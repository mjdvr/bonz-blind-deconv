# import essential libraries
import cv2 as cv
from matplotlib import pyplot as plt

# import and show image
f = cv.imread('./img/alphabet/a.png', 0)
plt.imshow(f, cmap = 'gray')
plt.show()