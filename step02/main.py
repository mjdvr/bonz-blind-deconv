import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve as fftc

def matlab_style_gauss2D(shape=(5,5),sigma=1):
    # source: https://stackoverflow.com/a/17201686
    """ 2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma]) """
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

def conv_prod(a, b, c):
    """ this function takes two arguments: a and b, and returns the convolution product of the two """
    return np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b))

def loop(nr_iter=3):
    for i in range(nr_iter):
        arr1 = np.random.rand(3,3)
        arr2 = np.random.rand(3,3)
        print(np.sqrt(np.mean((arr1-arr2)**2)))

def f_loop(f, g, c, nr_iter):
    """ this function takes an initial f, g and c, and returns the new f after 'nr_iter' RL iterations """
    if g.shape != f.shape:
        # reshape and center the psf to match the size of the image to make convolution possible
        g = g_reshape(g, f.shape)

    for i in range(nr_iter):
        print(i, f.shape)
        f = fftc(c/fftc(f,g,'same'),np.flip(g),'same')*f
        f = normalize_complex_arr(f)
    plt.imshow(f.real, cmap="gray")
    plt.show()
    return f.real

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
psf = matlab_style_gauss2D(shape=(15,15), sigma=1)
image = conv2(f, psf, 'same')

f_loop(image,psf,image,10)