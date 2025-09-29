import cv2
import numpy as np

# img = cv2.imread('images/ehidna.png')
# print(img.shape)
#
# # img = cv2.resize(img, (800, 800))
# img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
# # img = cv2.rotate(img, cv2.ROTATE_180)
# # img = cv2.flip(img, 0)
# # img = cv2.GaussianBlur(img, (9, 9), 15)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Canny(img, 100, 100)
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# img = cv2.erode(img, kernel, iterations=1)
#
#
#
# cv2.imshow('tea', img)
# cv2.imshow('tea', img[150:300, 200:350])

# video = cv2.VideoCapture('video/echidna_video.mp4')
video = cv2.VideoCapture(0)
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
