import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# make the plots more comprehensive
fig, ax = plt.subplots(2, 10, figsize=(18, 7))

def matlab_style_gauss2D(shape=(5,5),sigma=1):
    # source: https://stackoverflow.com/a/17201686
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def g_reshape(g, goal):
    """ this function take the psf and reshapes it to a size that matches that of the image """
    x_pad_affix = int(np.floor((goal[0]-g.shape[0])/2))
    x_pad_suffix = int(goal[0]-g.shape[0]-x_pad_affix)
    y_pad_affix = int(np.floor((goal[1]-g.shape[1])/2))
    y_pad_suffix = int(goal[1]-g.shape[1]-y_pad_affix)
    g = np.pad(g, ((x_pad_affix, x_pad_suffix),(y_pad_affix, y_pad_suffix)), 'constant')
    return g


def normalize_complex_arr(a):
    # source: https://stackoverflow.com/a/41576956
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()

def conv_prod(a, b):
    """ this function takes two arguments: a and b, and returns the convolution product of the two """
    return np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b))

def f_loop(f, g, c, nr_iter):
    """ this function takes an initial f, g and c, and returns the new f after 'nr_iter' RL iterations """
    if g.shape != f.shape:
        # reshape and center the psf to match the size of the image to make convolution possible
        g = g_reshape(g, f.shape)

    for i in range(nr_iter):
        f = conv_prod(c/conv_prod(f,g),np.flip(g))*f
        f = normalize_complex_arr(f)
        ax[0,i].imshow(f.real, cmap = 'gray')
    #plt.show()
    return f.real

def g_loop(f, g, c, nr_iter):
    """ this function takes an initial f, g and c, and returns the new g after 'nr_iter' RL iterations """
    if g.shape != f.shape:
        # reshape and center the psf to match the size of the image to make convolution possible
        g = g_reshape(g, f.shape)

    for i in range(nr_iter):
        g = conv_prod(c/conv_prod(g,f),np.flip(f))*g
        g = normalize_complex_arr(g)
        ax[1,i].imshow(g.real, cmap = 'gray')
    #plt.show()
    return g.real

def RL_decon(init_guess=1, nr_iter=5):
    a = init_guess
    return a

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
psf = matlab_style_gauss2D(shape=(5,5), sigma=1)
image = conv2(f, psf, 'same')

psf = matlab_style_gauss2D(shape=(2,50), sigma=10)
#psf = np.ones((100, 100)) / 25
f_loop(image,psf,image,10)
g_loop(image,psf,image,10)
plt.show()