import cv2
import csv
import numpy as np
from utils.tracking_utils import BallTracker, PlayerDetector, PingPongUmpire

def load_ground_truth(csv_path):
    """Loads the CSV and creates a tracking dictionary for each event."""
    gt_events = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_events.append({
                    'start': int(row['Start']),
                    'end': int(row['End']),
                    'name': row['Name'],
                    'detected_predictions': []
                })
    except Exception as e:
        print(f"Warning: Could not load CSV. {e}")
    return gt_events

def draw_ui_overlay(img, text, position, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    overlay = img.copy()
    (w, h), _ = cv2.getTextSize(text, font, 1.0, 2)
    cv2.rectangle(overlay, (position[0]-10, position[1]-h-20), (position[0]+w+10, position[1]+10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, position, font, 1.0, color, 2, cv2.LINE_AA)

def main():
    #VIDEO_PATH = 'data/dataset_labelise/video_simple/video_simple.mp4'
    #CSV_PATH = 'data/dataset_labelise/video_simple/video_simple.mp4.csv'
    
    VIDEO_PATH = 'data/dataset_labelise/video_bastien/video2.mp4'
    CSV_PATH = 'data/dataset_labelise/video_bastien/video2.mp4.csv'

    gt_events = load_ground_truth(CSV_PATH)
    false_positives = 0 

    win_name = "Ping Pong AI Analyst - Evaluation Mode"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    tracker = BallTracker(window_name=win_name)
    tracker.setup_trackbars()
    umpire = PingPongUmpire()
    detector = PlayerDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    display_msg, marker, timer = "", None, 0

    print("Starting Evaluation... Press 'q' to stop early and see results.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.resize(frame, (1080, int(frame.shape[0] * (1080/frame.shape[1]))))
        frame = detector.process(frame)
        frame, ball, _, is_real = tracker.process(frame)

        if ball and is_real:
            result = umpire.update(ball[0], ball[1])
            
            if result:
                display_msg, marker = result
                timer = 40
                
                matched_to_gt = False
                for gt in gt_events:
                    if gt['start'] <= frame_idx <= gt['end']:
                        gt['detected_predictions'].append(display_msg)
                        matched_to_gt = True
                        break
                
                if not matched_to_gt:
                    false_positives += 1

        if timer > 0:
            clr = (0, 255, 0) if "PADDLE" in display_msg else (0, 165, 255)
            if "DOUBLE" in display_msg or "FAULT" in display_msg: clr = (0, 0, 255)
            
            debug_text = f"Frame {frame_idx}: {display_msg}"
            draw_ui_overlay(frame, debug_text, (50, 80), clr)
            
            if marker:
                cv2.drawMarker(frame, marker, clr, cv2.MARKER_TILTED_CROSS, 30, 3)
            timer -= 1

        for i in range(1, len(tracker.pts)):
            if tracker.pts[i-1] and tracker.pts[i]:
                cv2.line(frame, tracker.pts[i-1], tracker.pts[i], (0, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("           TRACKER PERFORMANCE METRICS")
    print("="*50)
    
    detected_count = 0
    
    for gt in gt_events:
        preds = gt['detected_predictions']
        if len(preds) > 0:
            detected_count += 1
            unique_preds = list(set(preds))
            print(f"[Frames {gt['start']:03d}-{gt['end']:03d}] {gt['name']:<25} : DETECTED \t(Umpire said: {unique_preds})")
        else:
            print(f"[Frames {gt['start']:03d}-{gt['end']:03d}] {gt['name']:<25} : MISSED")

    total_gt = len(gt_events)
    accuracy = (detected_count / total_gt) * 100 if total_gt > 0 else 0
    
    print("-" * 50)
    print(f"Accuracy (True Positives): {detected_count} / {total_gt} events found ({accuracy:.1f}%)")
    print(f"False Positives (Ghosts):  {false_positives} events triggered outside CSV windows")
    print("=" * 50 + "\n")

if __name__ == "__main__": 
    main()