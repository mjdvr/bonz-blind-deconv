import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# make the plots more comprehensive
fig_rows, fig_cols = 2, 15
fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(18, 7))

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

def f_loop(f, g, c, nr_iter, show=0):
    """ this function takes an initial f, g and c, and returns the new f after 'nr_iter' RL iterations """
    if g.shape != f.shape:
        # reshape and center the psf to match the size of the image to make convolution possible
        g = g_reshape(g, f.shape)

    # make sure we take the right number of plots to show in the output
    iter_list = [0]
    if show == 1:
        interval = np.array_split(np.arange(nr_iter), fig_cols-1)
        for x in range(len(interval)):
            iter_list.append(interval[x][-1])

    for i in range(nr_iter):
        f = conv_prod(c/conv_prod(f,g),np.flip(g))*f
        f = normalize_complex_arr(f)
        if i in iter_list and show == 1:
            n = iter_list.index(i)
            ax[0,n].imshow(f.real, cmap = 'gray')
            ax[0,n].set_title(f'{i}')
    #plt.show()
    return f.real

def g_loop(f, g, c, nr_iter, show=0):
    """ this function takes an initial f, g and c, and returns the new g after 'nr_iter' RL iterations """
    if g.shape != f.shape:
        # reshape and center the psf to match the size of the image to make convolution possible
        g = g_reshape(g, f.shape)

    # make sure we take the right number of plots to show in the output
    iter_list = [0]
    if show == 1:
        interval = np.array_split(np.arange(nr_iter), fig_cols-1)
        for x in range(len(interval)):
            iter_list.append(interval[x][-1])

    for i in range(nr_iter):
        g = conv_prod(c/conv_prod(g,f),np.flip(f))*g
        g = normalize_complex_arr(g)
        if i in iter_list and show == 1:
            n = iter_list.index(i)
            ax[1,n].imshow(g.real, cmap = 'gray')
            ax[1,n].set_title(f'{i}')
    #plt.show()
    return g.real

def RL_decon(image, psf, nr_iter=10, show=1):
    # make sure we take the right number of plots to show in the output
    iter_list = [0]
    if show == 1:
        interval = np.array_split(np.arange(nr_iter), fig_cols-1)
        for x in range(len(interval)):
            iter_list.append(interval[x][-1])

    # source: https://stackoverflow.com/a/35259180
    latent_est = image
    psf_hat = np.flip(psf)
    for i in range(nr_iter):
        est_conv      = conv2(latent_est,psf,'same')
        relative_blur = image / est_conv
        error_est     = conv2(relative_blur,psf_hat,'same')
        latent_est    = latent_est * error_est
        if i in iter_list and show == 1:
            n = iter_list.index(i)
            ax[0,n].imshow(latent_est.real, cmap = 'gray')
            ax[1,n].set_title(f'{i}')
    return latent_est

def loop_loop(f, g, h, internal=5, nr_iter=100, show=0, **kwargs):
    i = 0
    f_init = f

    diff = []

    # make sure we take the right number of plots to show in the output
    interval = np.array_split(np.arange(nr_iter), fig_cols-1)
    iter_list = [0]
    for x in range(len(interval)):
        iter_list.append(interval[x][-1])

    while i < nr_iter:
        f = f_loop(f,g,f_init,internal)
        g = g_loop(f,g,f_init,internal)
        if i in iter_list:
            n = iter_list.index(i)
            ax[0,n].imshow(f.real, cmap = 'gray')
            ax[0,n].set_title(f'{i}')
            ax[1,n].imshow(g.real, cmap = 'gray')
            ax[1,n].set_title(f'{i}')
        diff.append(featureMatchCompare(f.real, h.real))
        i += 1
    print(diff)

def featureMatchCompare(img1, img2):
    img1 = cv.cvtColor(np.float32(img1), cv.COLOR_GRAY2RGB)
    img2 = cv.cvtColor(np.float32(img2), cv.COLOR_GRAY2RGB)
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None:
        matches = [0]
    else:
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
    return len(matches)

# import and show image
f = cv.imread('./img/alphabet/w.png', 0)
psf = matlab_style_gauss2D(shape=(15,15), sigma=1)
image = conv2(f, psf, 'same')

psf = matlab_style_gauss2D(shape=(15,15), sigma=3)
#psf = np.ones((100, 100)) / 25
#f_loop(image,psf,image,1000)
#g_loop(image,psf,image,1000)
#RL_decon(image,psf,100)
loop_loop(f=image, g=psf, h=image, internal=3, nr_iter=100)
plt.show()