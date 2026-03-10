import cv2
from utils.tracking_utils import TableDetector, BallTracker, PlayerDetector

def main():
    WINDOW_NAME = "Smart Tracker (Live)"
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # try:
    #     table_detector = TableDetector('data/table_base.png')
    # except Exception as e:
    #     print(e)
    #     return

    ball_tracker = BallTracker(buffer_size=32, max_jump_dist=200, window_name=WINDOW_NAME)
    ball_tracker.setup_trackbars()

    player_detector = PlayerDetector()

    stream = cv2.VideoCapture(0)

    while True:
        ret, frame = stream.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        frame = player_detector.process(frame)
        #frame = table_detector.process(frame)
        frame = ball_tracker.process(frame)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()