import cv2
import numpy as np

f = np.zeros((400,600,3), np.uint8)
f[:] = (220, 245, 245)

img = cv2.imread('images/1.png')
img = cv2.resize(img, (120,150))

qr = cv2.imread('images/qr.png')
qr = cv2.resize(qr, (100,100))

cv2.rectangle(f,(10,10),(f.shape[1]-10,f.shape[0]-10),(115,134,134),4)
f[30: 30 + img.shape[0],30: 30 + img.shape[1]] = img
cv2.putText(f, "Mykyta Ruban", (img.shape[1] + 50, img.shape[0]//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
cv2.putText(f,"Computer Vision Student", (img.shape[1] + 50, img.shape[0]//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 84, 84), 2)
cv2.putText(f, "Email: nikitosruban007@gmail.com", (img.shape[1] + 50, img.shape[0]//2 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (4, 16, 182), 1)
cv2.putText(f, "Phone: +380996232602", (img.shape[1] + 50, img.shape[0]//2 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (4, 16, 182), 1)
cv2.putText(f, "12/11/2009", (img.shape[1] + 50, img.shape[0]//2 + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 16, 182), 1)
f[244:344, 460:560] = qr
cv2.putText(f, "OpenCV Business Card", (img.shape[1] + 5, img.shape[0]//2 + 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imwrite("business_card.png", f)


cv2.imshow('image', f)
cv2.waitKey(0)