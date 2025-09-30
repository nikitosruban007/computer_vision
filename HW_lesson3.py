import cv2

img = cv2.imread('images/me.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
cv2.rectangle(img, (390, 170), (600, 468), (0, 32, 191), 2)
cv2.putText(img, "Ruban Mykyta", (405, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 32, 191), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
