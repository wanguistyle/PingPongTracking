import cv2
from utils.tracking_utils import TableDetector, BallTracker

def main():
    VIDEO_PATH = 'data/videos/ping_pong_match_1.mov'
    WINDOW_NAME = "Smart Tracker (Video)"
    TARGET_WIDTH = 800
    LOOP_VIDEO = True

    # FIX 1: Allow the window to be resized properly
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        table_detector = TableDetector('data/table_base.png')
    except Exception as e:
        print(e)
        return

    ball_tracker = BallTracker(buffer_size=32, window_name=WINDOW_NAME)
    ball_tracker.setup_trackbars()

    stream = cv2.VideoCapture(VIDEO_PATH)
    if not stream.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    while True:
        ret, frame = stream.read()

        if not ret:
            if LOOP_VIDEO:
                stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        scale = TARGET_WIDTH / frame.shape[1]
        height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (TARGET_WIDTH, height))

        # FIX 2: Force the window to match the new frame size
        cv2.resizeWindow(WINDOW_NAME, TARGET_WIDTH, height)

        frame = table_detector.process(frame)
        frame = ball_tracker.process(frame)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()