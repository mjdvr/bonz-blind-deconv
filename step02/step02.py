import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# make the plots more comprehensive
fig, ax = plt.subplots(1, 5, figsize=(10, 7))

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

def conv_prod(a, b):
    # this function takes two arguments: a and b, and returns the convolution product of the two
    return np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b))

def f_loop(f, g, c, nr_iter):
    # this function takes an initial f, g and c, and returns the new f after 'nr_iter RL iterations
    for i in range(nr_iter):
        f = conv_prod(c/conv_prod(f,g),np.flip(g))*f
        ax[i].imshow(f.real, cmap = 'gray')
    plt.show()
    return f.real

def g_loop(f, g, c, nr_iter):
    # this function takes an initial f, g and c, and returns the new g after 'nr_iter' RL iterations
    for i in range(nr_iter):
        g = conv_prod(c/conv_prod(g,f),np.flip(f))*g
        ax[i].imshow(g.real, cmap = 'gray')
    plt.show()
    return g.real

def RL_decon(init_guess=1, nr_iter=5):
    a = init_guess
    return a

# import and show image
f = cv.imread('./img/alphabet/q.png', 0)
psf = matlab_style_gauss2D(shape=(100,100), sigma=1)
image = conv2(f, psf, 'same')

""" ax[0,0].imshow(f, cmap = 'gray')
ax[0,0].set_title("source image")

# convolute the source/clean image with this mask, and show in its subplot
ax[1,0].imshow(image, cmap='gray')
ax[1,0].set_title("convolved")

# deconvolve with the richardson_lucy module of skimage, using a different psf
psf = matlab_style_gauss2D(sigma=1)
deconvolved = restoration.richardson_lucy(image, psf, num_iter=1000, clip=True)
ax[0,1].imshow(deconvolved, cmap='gray')
ax[0,1].set_title("RL 1000")

# deconvolve with the richardson_lucy module of skimage
deconvolved = restoration.richardson_lucy(image, psf, num_iter=2500, clip=True)
ax[1,1].imshow(deconvolved, cmap='gray')
ax[1,1].set_title("RL 2500")

# show the plots
plt.show() """

psf = matlab_style_gauss2D(shape=(100,100), sigma=5)
#psf = np.ones((100, 100)) / 100
print(f_loop(f,psf,image,5))
#print(g_loop(f,psf,image,5)[48:53,48:53])