from skimage.io import imread, imshow, imsave
from skimage.util import img_as_float64
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import numpy as np
from task2_testdata import utils
import sys
from matplotlib import pyplot as plt

def balloon_force(x,y):

    # tangent vector
    dx, dy = np.gradient(x), np.gradient(y)
    # normal vector
    dx, dy = -dy, dx
    # unit normal vector
    dx_unit = dx/np.hypot(dx,dy)
    dy_unit = dy/np.hypot(dx,dy)

    return dx_unit, dy_unit

def reparametrization(x, y):

    x, y = np.hstack((x,x[0])), np.hstack((y,y[0]))
    N = x.shape[0]
    snake = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

    # Distance array, where d[i] – distance to point x[i]y[i] from point x[0]y[0]
    dist = np.hstack((0, np.cumsum(np.sqrt(np.sum((snake[1:N, :] - snake[0:N - 1, :]) ** 2, axis=1)))))

    x = np.interp(np.linspace(0, dist[N - 1], N-1), dist, x.flatten())
    y = np.interp(np.linspace(0, dist[N - 1], N-1), dist, y.flatten())

    return x, y


def active_contours(input_image, initial_snake, output_image, alpha, beta, tau, w_line, w_edge, kappa_1, kappa_2, sigma=2, re_param=100, iter_param=10000):
    """Active contour model

    Parameters
    ----------
    input_image: string
        Path of the input image
    initial_snake: 2-D sequence of floats
        Path of the initial snake txt file
    output_image: string, optional
        path of the output image
    alpha: float
        Contour elasticity parameter
    beta: float
        Contour stiffness parameter
    tau: float
        Time step (Snake speed) parameter
    w_line: float
        Line potential weight
    w_edge: float
        Edge potential weight
    kappa_1: float
        Balloon force parameter, where kappa_1 sign controls inflate or deflate
        note: |kappa_1| < |kappa_2| <1
    kappa_2: float
        External force parameter
        note: |kappa_1| < |kappa_2| <1
    sigma: float
        Gaussian filter parameter
    re_param: int
        Curve reparametrization frequency
    iter_param: int
        Number of iterations

    Returns
    -------
    snake: 2-D sequence of floats
        Contour
    """
    alpha, beta, tau, w_line, w_edge, kappa_1, kappa_2, sigma, re_param, iter_param = \
    float(alpha), float(beta), float(tau), float(w_line), float(w_edge), float(kappa_1),\
    float(kappa_2), float(sigma), int(re_param), int(iter_param)
    image = img_as_float64(imread(input_image, as_gray=True))
    snake = np.loadtxt(initial_snake)

    # Contour points
    x, y = snake[:, 0], snake[:, 1]
    # Number of contour points
    n = snake.shape[0]

    # A – Euler equation matrix, where c_n - the n order contour derivative
    c_2 = np.roll(np.eye(n), -1, axis=1) - 2*np.eye(n) + np.roll(np.eye(n), -1, axis=0)
    c_4 = np.roll(np.eye(n), -2, axis=1) - 4*np.roll(np.eye(n), -1, axis=1) + 6*np.eye(n) - 4*np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -2, axis=0)
    A = -alpha*c_2 + beta*c_4

    # (I-tau*A) inverse matrix
    A_inv = np.linalg.inv(np.eye(n) + tau*A)

    # Gaussian derivative filter
    # With respect to X
    dx = gaussian_filter(image, sigma=sigma, order=[1,0], output=np.float64, mode='nearest')
    # With respect to Y
    dy = gaussian_filter(image, sigma=sigma, order=[0,1], output=np.float64, mode='nearest')

    # Potential (for External energy)
    Potential_edge = - (dx**2 + dy**2)
    Potential_line = - gaussian_filter(image, sigma=sigma, order=[0,0], output=np.float64, mode='nearest')
    Potential = - w_line*Potential_line - w_edge*Potential_edge
    # Poltential interpolation
    Potential_interp = RectBivariateSpline(np.arange(Potential.shape[1]), np.arange(Potential.shape[0]), Potential.T)

    # Main Loop
    j=1
    while j < iter_param:

        # Potential gradient
        P_x, P_y = Potential_interp(x, y, dx=1, grid=False), Potential_interp(x, y, dy=1, grid=False)
        n_x, n_y = balloon_force(x, y)
        F_ext_x = kappa_1*n_x - kappa_2*P_x/np.hypot(P_x, P_y)
        F_ext_y = kappa_1*n_y - kappa_2*P_y/np.hypot(P_x, P_y)

        x = A_inv.dot(x+tau*F_ext_x)
        y = A_inv.dot(y+tau*F_ext_y)

        # Reparametrization
        if j % re_param == 0:
            x, y = reparametrization(x, y,)

        j += 1

    new_snake = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    utils.save_mask(output_image, new_snake, image)

    return new_snake


if __name__ == '__main__':
    if len(sys.argv) > 2:
        globals()[sys.argv[1]](*[sys.argv[i] for i in range(2, len(sys.argv))])