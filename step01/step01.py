# import essential libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# make the plots more comprehensive
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
ax[0].imshow(f, cmap = 'gray')
ax[0].set_title("source image")

# create fourier transform of the loaded image
f_fourier = np.fft.fft2(f)
f_shift = np.fft.fftshift(f_fourier)
f_mag_spec = 20*np.log(np.abs(f_shift))
ax[1].imshow(f_mag_spec, cmap = 'gray')
ax[1].set_title("magnitude spectrum")
plt.show()