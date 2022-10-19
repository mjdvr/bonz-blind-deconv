# import essential libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa

# make the plots more comprehensive
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
ax[0,0].imshow(f, cmap = 'gray')
ax[0,0].set_title("source image (f)")

# create fourier transform of the loaded image
f_fourier = np.fft.fft2(f)
f_shift = np.fft.fftshift(f_fourier)
f_mag_spec = 20*np.log(np.abs(f_shift))
ax[1,0].imshow(f_mag_spec, cmap = 'gray')
ax[1,0].set_title("magnitude spectrum (f)")

# generate a gaussian transform mask, and create the distorted image
aug_gauss = iaa.AdditiveGaussianNoise(scale=(10, 60), seed=1)
b = aug_gauss(image=f)
ax[0,1].imshow(b, cmap = 'gray')
ax[0,1].set_title("distorted image (b)")

# foerier transform, shift and create a magnitude spectrum as above
b_fourier = np.fft.fft2(b)
b_shift = np.fft.fftshift(b_fourier)
b_mag_spec = 20*np.log(np.abs(b_shift))
ax[1,1].imshow(b_mag_spec, cmap = 'gray')
ax[1,1].set_title("magnitude spectrum (b)")

# show the plots
plt.show()