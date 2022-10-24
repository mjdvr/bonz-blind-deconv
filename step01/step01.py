# import essential libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# make the plots more comprehensive
fig, ax = plt.subplots(2, 7, figsize=(24, 7))

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
ax[0,0].imshow(f, cmap = 'gray')
ax[0,0].set_title("source image (f)")

# create fourier transform of the loaded image
f_fourier = np.fft.fft2(f)
f_shift = np.fft.fftshift(f_fourier)
f_mag_spec = np.log(np.abs(f_shift))
ax[1,0].imshow(f_mag_spec, cmap = 'gray')
ax[1,0].set_title("magnitude spectrum (f)")

# generate a gaussian transform mask, and create the distorted image
aug_gauss = iaa.AdditiveGaussianNoise(scale=(10, 60), seed=1)
b = aug_gauss(image=f)
ax[0,1].imshow(b, cmap = 'gray')
ax[0,1].set_title("distorted image (b)")

# fourier transform, shift and create a magnitude spectrum as above
b_fourier = np.fft.fft2(b)
b_shift = np.fft.fftshift(b_fourier)
b_mag_spec = np.log(np.abs(b_shift))
ax[1,1].imshow(b_mag_spec, cmap = 'gray')
ax[1,1].set_title("magnitude spectrum (b)")

# simply dividing the distorted image by the original one yields the mask the function generated
    # NOTE: this only works when the distorted image is not 0
h_fourier = b_fourier/f_fourier
    # inverse fourier transform of the mask
h = np.fft.ifft2(h_fourier)
ax[0,2].imshow(abs(h), cmap = 'gray')
ax[0,2].set_title("gaussian mask (h)")
    # get the shifted spectrum,a dn generate the corresponding magnitude spectrum
h_shift = np.fft.ifftshift(h_fourier)
h_mag_spec = np.log(abs(h_shift))
ax[1,2].imshow(h_mag_spec, cmap = 'gray')
ax[1,2].set_title("magnitude spectrum (h)")

# Wiener deconvolution
    # first get the shifted mask, and select the middle (the most relevant part)
psf = abs(h_shift)[48:53,48:53]
    # show the previously convoluted image
ax[0,3].imshow(b, cmap='gray')
ax[0,3].set_title("distorted image (b)")
    # apply the wiener deconvolution to (b) with the middle 5x5 kernel from the deduced psf (h)
deconvolved = restoration.wiener(b, psf, 1100, clip=False)
ax[1,3].imshow(deconvolved, cmap='gray')
ax[1,3].set_title("Wiener (b)")

    # convolute the source/clean image with this mask, and show in its subplot
image = conv2(f, psf, 'same')
ax[0,4].imshow(image, cmap='gray')
ax[0,4].set_title("convolved (clean)")
    # deconvolve with the wiener module of skimage
deconvolved = restoration.wiener(image, psf, 1100, clip=False)
ax[1,4].imshow(deconvolved, cmap='gray')
ax[1,4].set_title("Wiener (clean)")

# Richardson-Lucy deconvolution
    # we first show the same custom convoluted image
ax[0,5].imshow(b, cmap='gray')
ax[0,5].set_title("distorted image (b)")
    # apply the richardson_lucy deconvolution to (b) with same kernel as above
deconvolved = restoration.richardson_lucy(b, psf, num_iter=100, clip=True)
ax[1,5].imshow(deconvolved, cmap='gray')
ax[1,5].set_title("RL (b)")

    # convolute the source/clean image with this mask, and show in its subplot
ax[0,6].imshow(image, cmap='gray')
ax[0,6].set_title("convolved (clean)")
    # deconvolve with the richardson_lucy module of skimage
deconvolved = restoration.richardson_lucy(image, psf, num_iter=2500, clip=True)
ax[1,6].imshow(deconvolved, cmap='gray')
ax[1,6].set_title("RL (clean)")

# show the plots
plt.show()