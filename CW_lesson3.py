import cv2
import numpy as np

img = np.zeros((500, 400, 3), np.uint8)
# img[:] = (94, 235, 52)

# img[100:150, 200:250] = (94, 235, 52)

cv2.rectangle(img, (100, 100), (200, 200), (94, 235, 52), 1)
cv2.line(img, (100, 100), (200, 200), (94, 235, 52), 1)
cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (94, 235, 52), 1)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (94, 235, 52), 1)
cv2.circle(img, (200, 200), 30, (94, 235, 52), -1)
cv2.putText(img, "Ishooo", (150, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (94, 235, 52), 2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()