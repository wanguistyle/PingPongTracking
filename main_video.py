import cv2
import numpy as np
from utils.tracking_utils import BallTracker, PlayerDetector, PingPongUmpire

def draw_ui_overlay(img, text, position, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    overlay = img.copy()
    (w, h), _ = cv2.getTextSize(text, font, 1.0, 2)
    cv2.rectangle(overlay, (position[0]-10, position[1]-h-20), (position[0]+w+10, position[1]+10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, position, font, 1.0, color, 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture('data/videos/pingpong_videos/IMG_2193.MOV')
    win_name = "Ping Pong AI Analyst"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    tracker = BallTracker(window_name=win_name)
    tracker.setup_trackbars()
    umpire = PingPongUmpire()
    detector = PlayerDetector()

    display_msg, marker, timer = "", None, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            umpire.reset_rally()
            continue

        frame = cv2.resize(frame, (1080, int(frame.shape[0] * (1080/frame.shape[1]))))
        frame = detector.process(frame)
        frame, ball, _, is_real = tracker.process(frame)

        if ball and is_real:
            result = umpire.update(ball[0], ball[1])
            if result:
                display_msg, marker = result
                timer = 40

        if timer > 0:
            clr = (0, 255, 0) if "PADDLE" in display_msg else (0, 165, 255)
            if "DOUBLE" in display_msg: clr = (0, 0, 255)
            draw_ui_overlay(frame, display_msg, (50, 80), clr)
            if marker:
                cv2.drawMarker(frame, marker, clr, cv2.MARKER_TILTED_CROSS, 30, 3)
            timer -= 1

        # Draw trajectory trail
        for i in range(1, len(tracker.pts)):
            if tracker.pts[i-1] and tracker.pts[i]:
                cv2.line(frame, tracker.pts[i-1], tracker.pts[i], (0, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()