import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = []
object_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            object_detected = True

            cv2.drawContours(frame, [cnt], 0, (0, 255, 4), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cx, cy), 5, (255, 123, 4), -1)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
                cv2.putText(frame, "Object detected!", (430, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # points.append((cx, cy))
                # for i in range(1, len(points)):
                #     if points[i - 1] is None or points[i] is None:
                #         continue
                #     cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)
    if not object_detected:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
        cv2.putText(frame, "Object not detected!", (375, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('hsv', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()