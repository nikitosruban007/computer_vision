import cv2
import numpy as np


img = cv2.imread('images/img.jpg')
scale = 2
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
final = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([160, 0, 0])
upper_red = np.array([179, 255, 255])

lower_green = np.array([44, 82, 110])
upper_green = np.array([98, 255, 255])

lower_yellow = np.array([0, 0, 0])
upper_yellow = np.array([40, 255, 255])

lower_blue = np.array([80, 102, 0])
upper_blue = np.array([150, 255, 255])

mask_red = cv2.inRange(img, lower_red, upper_red)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)
mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)
img = cv2.bitwise_or(img, img, mask=mask_total)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        hsl = np.mean(img.reshape(-1, 3), axis=0)
        hue, s, l = hsl
        color = ''
        if 0 < hue < 10 or 160 < hue< 179:
            color = 'Red'
        elif 26 < hue < 35:
            color = 'Yellow'
        elif 101 < hue < 130:
            color = 'Blue'
        elif 35 < hue < 85:
            color = 'Green'
        else:
            color = 'IDK'
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        M = cv2.moments(cnt)

        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        shape = ''

        if len(approx) == 3:
            shape = 'Triangle'
        elif len(approx) == 4:
            shape = 'Rectangle'
        elif len(approx) >= 8 and len(approx) != 12:
            shape = 'Oval'
        elif len(approx) == 12:
            shape = "Star"
        else:
            shape = 'Idk'


        cv2.circle(final, (cX, cY), 5, (255, 255, 255), -1)
        cv2.drawContours(final, [cnt], -1, (255, 255, 255), 2)
        cv2.putText(final, f'Coordinates: x = {x}, y = {y}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0, 0), 1)
        cv2.putText(final, f'Figure: {shape}',(x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0, 0), 1)
        cv2.putText(final, f'S = {area}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(final, f'{color}', (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

cv2.imshow('img', img)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()