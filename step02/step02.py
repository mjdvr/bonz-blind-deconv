import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# make the plots more comprehensive
fig, ax = plt.subplots(2, 2, figsize=(10, 7))

def matlab_style_gauss2D(shape=(5,5),sigma=0.5):
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

def RL_iter(value):
    value = value+1
    return value

def RL_decon(init_guess, nr_iter=5):
    a = init_guess
    for i in range(nr_iter):
        print(RL_iter(i))
    return a

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
ax[0,0].imshow(f, cmap = 'gray')
ax[0,0].set_title("source image")

psf = matlab_style_gauss2D()
image = conv2(f, psf, 'same')
# convolute the source/clean image with this mask, and show in its subplot
ax[1,0].imshow(image, cmap='gray')
ax[1,0].set_title("convolved")

# deconvolve with the richardson_lucy module of skimage
deconvolved = restoration.richardson_lucy(image, psf, num_iter=1000, clip=True)
ax[0,1].imshow(deconvolved, cmap='gray')
ax[0,1].set_title("RL 1000")

# deconvolve with the richardson_lucy module of skimage
deconvolved = restoration.richardson_lucy(image, psf, num_iter=2500, clip=True)
ax[1,1].imshow(deconvolved, cmap='gray')
ax[1,1].set_title("RL 2500")

# show the plots
#plt.show()

RL_decon(1)