import cv2

cap = cv2.VideoCapture(0)

ret, frame_prev_color = cap.read()

frame_prev = cv2.cvtColor(frame_prev_color, cv2.COLOR_BGR2GRAY)
frame_prev = cv2.GaussianBlur(frame_prev, (5, 5), 0)

while True:
    ret, frame_curr_color = cap.read()

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_copy = frame_curr_color.copy()

    frame_curr_gray = cv2.cvtColor(frame_curr_color, cv2.COLOR_BGR2GRAY)
    frame_curr = cv2.GaussianBlur(frame_curr_gray, (5, 5), 0)

    diff = cv2.absdiff(frame_prev, frame_curr)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    frame_prev = frame_curr

    cv2.imshow('Difference', frame_copy)


cap.release()
cv2.destroyAllWindows()