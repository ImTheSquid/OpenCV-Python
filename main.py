import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.line(frame, (0, 0), (300, 300), (255, 255, 255), 10)
    cv2.rectangle(frame,(20,20),(350,200),(0,0,200),5)
    cv2.putText(frame, "Hello", (0,300), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

    # Display the resulting frame
    cv2.imshow('Normal', frame)
    cv2.imshow('Grayscale',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
