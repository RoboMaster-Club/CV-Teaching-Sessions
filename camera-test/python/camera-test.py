import cv2

cap = cv2.VideoCapture(0)
pause = False
cv2.namedWindow("Camera")
while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        cv2.imshow("Camera", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
        elif c & 0xFF == ord(' '):
            pause = not pause

cap.release()
cv2.destroyAllWindows()
