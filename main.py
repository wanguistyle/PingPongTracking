import cv2
from utils.tracking_utils import TableDetector, BallTracker

def main():
    WINDOW_NAME = "Smart Tracker"
    cv2.namedWindow(WINDOW_NAME)

    try:
        table_detector = TableDetector('data/table_base.png')
    except Exception as e:
        print(e)
        return

    ball_tracker = BallTracker(buffer_size=32, window_name=WINDOW_NAME)
    ball_tracker.setup_trackbars()

    stream = cv2.VideoCapture(0)

    while True:
        ret, frame = stream.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        frame = table_detector.process(frame)
        frame = ball_tracker.process(frame)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()