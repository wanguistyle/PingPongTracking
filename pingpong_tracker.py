import cv2
import numpy as np
from collections import deque

BUFFER_SIZE = 32
MIN_CIRCULARITY = 0.5  # 1.0 serait un cercle parfait => engendre des diffculés lorsque la balle va trop vite

pts = deque(maxlen=BUFFER_SIZE)

stream = cv2.VideoCapture(0)

cv2.namedWindow("Adjust Colors")
def nothing(x): pass

#Orange classique, peut être précisé 
cv2.createTrackbar("Lower Hue", "Adjust Colors", 5, 179, nothing)
cv2.createTrackbar("Upper Hue", "Adjust Colors", 25, 179, nothing)
cv2.createTrackbar("Lower Sat", "Adjust Colors", 130, 255, nothing)
cv2.createTrackbar("Upper Sat", "Adjust Colors", 255, 255, nothing)
cv2.createTrackbar("Lower Val", "Adjust Colors", 100, 255, nothing)
cv2.createTrackbar("Upper Val", "Adjust Colors", 255, 255, nothing)

print("Tracking: Orange Color + Round Shape")

while True:
    ret, frame = stream.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Get Sliders
    l_h = cv2.getTrackbarPos("Lower Hue", "Adjust Colors")
    u_h = cv2.getTrackbarPos("Upper Hue", "Adjust Colors")
    l_s = cv2.getTrackbarPos("Lower Sat", "Adjust Colors")
    u_s = cv2.getTrackbarPos("Upper Sat", "Adjust Colors")
    l_v = cv2.getTrackbarPos("Lower Val", "Adjust Colors")
    u_v = cv2.getTrackbarPos("Upper Val", "Adjust Colors")

    mask = cv2.inRange(hsv, (l_h, l_s, l_v), (u_h, u_s, u_v))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        x, y, w, h = cv2.boundingRect(c)
        
        aspect_ratio = float(w) / h
        if area > 300 and circularity > MIN_CIRCULARITY and 0.6 < aspect_ratio < 1.4:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            
            coord_text = f"({cX}, {cY})"
            cv2.putText(frame, coord_text, (cX - 20, cY - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None: continue
        thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Smart Tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()