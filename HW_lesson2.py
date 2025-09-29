import cv2
import numpy as np


img = cv2.imread('images/me.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 40, 40)
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)

img2 = cv2.imread('images/me2.jpg')
img2 = cv2.resize(img2, (img.shape[1] // 1, img.shape[0] // 1))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.Canny(img2, 40, 40)
kernel1 = np.ones((1, 1), np.uint8)
img2 = cv2.dilate(img2, kernel1, iterations=1)
img2 = cv2.erode(img2, kernel1, iterations=1)


cv2.imshow('1', img)
cv2.imshow('2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()