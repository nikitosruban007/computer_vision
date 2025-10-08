#P.S. ніхто б не захотів з групи сфотаться зі мною, тому просто фото з інету =(


import cv2

img = cv2.imread('images/OIP.webp')
scale = 2
img_copy_color = img.copy()
img_copy = img.copy()

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)


img_copy = cv2.equalizeHist(img_copy)

img_copy = cv2.Canny(img_copy, 190, 190)

contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 30:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f'x={x}, y={y}, S = {int(area)}'
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


cv2.imshow('contours', img)
cv2.imshow('copy contours', img_copy)
cv2.imshow("img", img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()