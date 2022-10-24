# step01
Folder in which the first steps are taken towards the end goal.
In this step:
 - &check; read image
 - &check; create a fourier transform of this image
 - &check; generate a transform mask (automated through the *imgaug* package)
 - &check; apply mask to image, thus creating a convoluted image
 - &check; extract the mask by dividing the fourier transforms
 - &check; deconvolve the image with a simple scheme (Wiener deconvolution)
 - &check; deconvolve the image with a slightly more complex scheme (RL deconvolution)