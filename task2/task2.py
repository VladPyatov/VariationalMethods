import sys
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.color import rgba2rgb
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib import pyplot as plt


def functional(img, kernel, real_img, alpha, noise = 0):
    """Tikhonov regularization functional
        | Az+noise-u |^2 + alpha*BTV[z]
    Parameters
    ----------
    img: 2-D array
        input image
    kernel: 2-D array
        kernel image
    real_img: string, optional
        deconvolved image
    alpha: float, optional
        BTV parameter
    noise_level: float, optional
        input image noise level - [0,255]

    Returns
    -------
    deConv: 1x3 array
        Tikhonov regularization functional value at z (deconvolved image) point
    """
    n_channels = img.shape[2]
    strides = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    # | Az+noise-u |^2
    error = np.zeros(img.shape)

    for ch in range(n_channels):
        error[:, :, ch] = convolve(real_img[:, :, ch], kernel, mode="nearest")
    error += noise-img

    # Functional
    J = np.sum(error ** 2, axis=(0, 1))

    # Extrapolation
    real_img = np.vstack((real_img[0, np.newaxis, :], real_img, real_img[-1, np.newaxis, :]))
    real_img = np.hstack((real_img[:, 0, np.newaxis], real_img, real_img[:, -1, np.newaxis]))

    # BTV computation
    BTV = np.zeros(n_channels)
    for x, y in strides:
        sums = np.roll(real_img, (x, y), axis=(0, 1)) - real_img
        BTV += np.sum(sums, axis=(0,1)) / np.hypot(x, y)

    return J + alpha * BTV

def contrasting(img):
    # Contrasting

    if len(img.shape) == 3:
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        U = -0.0999 * R - 0.3360 * G + 0.4360 * B
        V = 0.6150 * R - 0.5586 * G - 0.0563 * B
    else:
        Y = img

    new_y = np.ravel(Y.copy())
    new_y.sort()
    x_min = new_y[0]
    x_max = new_y[-1]

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (Y[i, j] - x_min) * 255 / (x_max - x_min)

    if len(img.shape) == 3:
        R = Y + 1.2803 * V
        G = Y - 0.2148 * U - 0.3805 * V
        B = Y + 2.1279 * U
        img = np.dstack((R, G, B))
    else:
        img = Y

    return img


def deconvolution(input_image, kernel, output_image, alpha=0.1, betta=0.5, mu=0.9, smart_step=False, noise_level=0., psnr_log=False):
    """Image deconvolution model

    Parameters
    ----------
    input_image: string
        Path of the input image
    kernel: string
        Path of the convolution kernel image
    output_image: string, optional
        path of the output image
    alpha: float, optional
        BTV parameter
    betta: float, optional
        Gradient step size
    mu: float, optional
        Nesterov gradient parameter
    smart_step: bool
        Make a "gradient" move only if functional value decreases
    noise_level: float
        The level of noise
        If = -1 => calculates automatically (not recommended)
    psnr_log: bool, optional
        PSNR per iteration plotting option

    Returns
    -------
    deConv: 2-D array
        Deconvolved image
    """

    img = imread(input_image)

    kernel = (imread(kernel, as_gray=True)*255).astype(np.float64)
    kernel *= 1 / kernel.sum()

    alpha = float(alpha)
    betta = float(betta)
    mu = float(mu)
    smart_step = bool(smart_step)
    noise_level = float(noise_level)
    psnr_log = bool(psnr_log)

    # Pre-processing
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = rgba2rgb(img) * 255
    elif len(img.shape) == 2:
        img = (img[:, :, np.newaxis]).astype(np.float64)
    else:
        img = img.astype(np.float64)

    n_channels = img.shape[2]

    if noise_level == -1:
        noise_level=np.zeros(img.shape[2])
        gauss = np.zeros(img.shape)
        for ch in range(n_channels):
            gauss[:,:,ch] = gaussian_filter(img[:,:,ch], sigma=1, order=[0, 0], output=np.float64, mode='nearest')
            noise_level[ch] = (median_filter(gauss[:, :, ch], size=5, output=np.float64, mode='nearest') - img[:,:,ch]).var()
        noise_level=np.sqrt(noise_level)
    else:
        noise_level = np.full(3, noise_level)


    # Deconvolved image
    deConv = img.copy()
    new_deConv = np.zeros(deConv.shape)

    # Functional gradient
    grad = np.zeros(deConv.shape)

    # Nesterov gradient parameter
    V = np.zeros(deConv.shape)

    strides = [(1, 0), (0, 1), (1, 1), (1, -1)]

    # PSNR/iteration array
    psnr = []

    best_func = functional(img, kernel, deConv, alpha)

    # Minimization
    for i in range(1, 1001):

        # Error derivative calculation

        for ch in range(n_channels):
            grad[:, :, ch] = convolve(deConv[:, :, ch] + mu * V[:, :, ch], kernel, mode="nearest") \
                             + noise_level[ch] - img[:, :, ch]
            grad[:, :, ch] = convolve(grad[:, :, ch], kernel, mode="nearest") * 2

        # BTV derivative calculation
        DBTV = np.zeros(img.shape)

        for x, y in strides:
            sgn = np.sign(np.roll(deConv + mu * V, (x, y), axis=(0, 1)) - deConv - mu * V)
            TV = np.roll(sgn, (-x, -y), axis=(0, 1)) - sgn
            DBTV += TV / np.hypot(x, y)

        # Gradient update
        grad += alpha * DBTV

        # Nesterov parameter update
        V = mu * V - betta * grad

        # Argument (image) update
        new_deConv = deConv + V


        if smart_step:
            new_func=functional(img,kernel,new_deConv,alpha)
            if new_func[0]<=best_func[0] and new_func[1]<=best_func[1] and new_func[2]<=best_func[2]:
                deConv = new_deConv
                best_func=new_func.copy()
        else:
            deConv = new_deConv

        psnr.append(20 * np.log10(255 / np.sqrt(np.sum((img - deConv) ** 2 / (3 * img.shape[0] * img.shape[1])))))

        if i > 10 and sum(np.abs(np.array(psnr[-10:]) - np.array(psnr[-11:-1]))) <= 0.001:
            break

    if psnr_log:
        print("Min PSNR:", min(psnr))
        print("PSNR per iterations:", psnr)
        plt.plot(psnr)
        plt.show()

    deConv = np.clip((deConv+noise_level).round(), 0, 255).astype(np.uint8)

    imsave(output_image, deConv)

    return deConv


if __name__ == '__main__':

    if len(sys.argv) > 2:
        globals()[sys.argv[1]](*[sys.argv[i] for i in range(2, len(sys.argv))])
