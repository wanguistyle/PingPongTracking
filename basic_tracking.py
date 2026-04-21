import cv2
import numpy as np

# --- 1. Load the video ---
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- 2. Define HSV color range for the ball ---
lower_color = np.array([8, 180, 180])
upper_color = np.array([18, 255, 255])

out = cv2.VideoWriter(
    "output_detected.mp4",  # Output file name
    cv2.VideoWriter_fourcc(*"mp4v"),  # Codec
    fps,  # Frames per second
    (frame_width, frame_height),  # Frame size
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. Convert frame to HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- 4. Create mask for the ball color ---
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # --- 5. Reduce noise ---
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, None, iterations=2)

    # --- 6. Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter small contours
        if cv2.contourArea(cnt) < 50:
            continue

        # Find the center and radius of the ball
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the detected ball
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 2, (0, 0, 255), -1)  # center dot

    out.write(frame)

    # --- 7. Show result ---
    cv2.imshow("Ball Detection", frame)

    # Slow down by increasing waitKey delay
    # slow_factor = 3
    # if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
    # break

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
