# Ping Pong AI Analyst

Ping Pong AI Analyst is a computer vision-based tool designed to track table tennis matches, analyze ball trajectories, and classify player strokes in real-time. By leveraging OpenCV for ball tracking and MediaPipe for pose estimation, this system acts as a virtual umpire and coach, evaluating match events and comparing them against ground-truth data to measure AI performance.

## Key Features

* **Robust Ball Tracking**: Uses HSV color filtering, contour detection, and momentum-based "coasting" to track the ping pong ball even when it drops out of frame for a few milliseconds.
* **Virtual Umpire Logic**: Analyzes ball trajectory peaks and directional reversals to detect table bounces, paddle impacts, and faults (e.g., double bounces).
* **Stroke Classification (Forehand vs. Backhand)**: Utilizes MediaPipe Pose to track the player's right wrist and elbow, classifying shots as either "COUP DROIT" (Forehand) or "REVERS" (Backhand) upon paddle impact.
* **Evaluation & Metrics**: Compares the AI's real-time detection against manually annotated CSV files (Ground Truth) to calculate Precision, Recall, Accuracy, False Positives, and False Negatives.
* **Interactive UI**: Provides an on-screen overlay displaying the current frame, detected impacts, bounding boxes, and historical ball trajectory lines.

## Project Structure

Based on the core files, the project is structured as follows:

* **`main.py`**: The primary evaluation script configured for the live stream. It live processes the AI logic.
* **`main_video.py`**: An alternate evaluation script configured for the dataset. Similar to `main.py` but includes different metric calculations (Precision/Recall vs. general Accuracy).
* **`utils/tracking_utils.py`**: The core engine containing the main classes:
    * `BallTracker`: Handles OpenCV image processing, HSV masking, and spatial tracking.
    * `PingPongUmpire`: Analyzes the tracked coordinates to determine the context of the bounce (Table vs. Paddle).
    * `PlayerDetector`: Wraps MediaPipe's Pose detection to analyze player biomechanics and classify stroke types.
* **`data/`** *(Directory)*: Expected to contain the video files (`.mp4`) and their corresponding annotation files (`.csv`).
* **`tests/`**: A place where to practice on new codes and technologies before puting it or not in utils

## Prerequisites & Installation

To run this project, you will need Python 3 installed along with several external libraries. You can install the required dependencies using `pip`:

```bash
pip install opencv-python numpy mediapipe