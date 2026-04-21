## Evaluation Methodology & Statistical Validation

To ensure the reliability of the Ping Pong AI Analyst, our evaluation methodology is split into two distinct testing protocols. The first rigidly benchmarks the **Spatial Tracking Accuracy** of the computer vision engine, while the second assesses the system's real-time **Temporal & Categorical Accuracy** (event classification).

### 1. Spatial Tracking Accuracy (`metrics.ipynb`)

To validate the system's ability to precisely locate the ball frame-by-frame, we benchmarked three models across three test videos (`video_immobile`, `video_lente`, `video_simple`) from the labelled dataset:

* **YOLO** (`best-2.pt`) — fine-tuned YOLO model, frame-by-frame inference via `ultralytics`
* **Computer Vision Tracker** (`BallTracker`) — advanced OpenCV tracker using HSV filtering, contour detection and a momentum-based "coasting" mechanism (from `utils/tracking_utils_2D.py`)
* **Basic HSV Tracker** — simple color detection via an orange HSV mask (`[5–15°, S>150, V>150]`) with no tracking logic

Each model produces a list of `[frame_idx, cx, cy, confidence]` tuples. These predictions are then aligned against the dense Ground Truth dataset (`annotations.csv`), which contains manually annotated bounding boxes whose centroids `(cx_gt, cy_gt)` are computed at evaluation time.

#### 1.1 Distance & Error Metrics
For every frame $i$ where the ball is annotated, we extract the model's predicted centroid $(cx_{pred}, cy_{pred})$ and compare it to the ground truth centroid $(cx_{gt}, cy_{gt})$. The primary spatial metric is the **Euclidean Distance (L2 Norm)**:

$$Distance_i = \sqrt{(cx_{pred} - cx_{gt})^2 + (cy_{pred} - cy_{gt})^2}$$

To aggregate these frame-by-frame distances into robust statistical indicators, we calculate the following regression metrics:

* **MAE (Mean Absolute Error):** Represents the average tracking error in pixels.
$$MAE = \frac{1}{N} \sum_{i=1}^{N} Distance_i$$

* **RMSE (Root Mean Square Error):** Heavily penalizes large tracking deviations or "jumps," making it an excellent indicator of tracking stability and consistency.
$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (Distance_i)^2}$$

* **Maximum Error:** The single largest pixel deviation recorded in the sequence, used to identify extreme outliers and tracking failures.

#### 1.2 Detection Rate
Because models occasionally lose the ball entirely due to motion blur or player occlusion, we also measure the **Detection Rate**—the percentage of annotated frames where the model successfully proposed a bounding box or centroid:

$$Detection\ Rate = \left( \frac{Frames\ Detected}{Total\ GT\ Frames} \right) \times 100$$

*By cross-referencing MAE, RMSE, and Detection Rate, we determine not only which model tracks the ball closest to physical reality, but which is the most reliable across varying lighting and speed conditions.*


---

### 2. Temporal & Categorical Event Matching (`main.py`)
While spatial tracking handles the *where*, the virtual umpire logic handles the *what* and *when*. We evaluate stroke classifications (Forehand vs. Backhand) by aligning real-time AI predictions against temporal CSV annotations.

#### 2.1 Confusion Matrix Components
Because the AI operates continuously, a predicted event is considered valid only if it falls within a human-annotated temporal window. Every event is categorized into one of four states:

* **True Positives (TP - *Vrais Positifs*):** The AI detected an impact within the `[Start, End]` temporal window **and** assigned the correct stroke class.
* **Classification Errors (*Erreurs de classe*):** The AI detected the impact at the correct time, but assigned the **wrong class** (e.g., predicted Forehand when GT was Backhand).
* **False Negatives (FN - *Faux Négatifs*):** A Ground Truth stroke occurred, but the AI failed to trigger any impact event.
* **False Positives (FP - *Faux Positifs*):** The AI triggered an impact outside of any valid ground-truth temporal window (phantom bounces/strokes).

#### 2.2 Global Performance Indicators
Using the raw counts from the matching strategy, we calculate standard classification metrics to evaluate the virtual umpire's reliability:

* **Recall (Sensitivity):** Measures the model's ability to find all relevant events without missing them.
$$Recall = \frac{TP}{Total\ GT\ Events} \times 100$$

* **Precision:** Measures the accuracy of the model's positive predictions. When the AI states a stroke happened, how often is it a real event?
$$Precision = \frac{TP}{TP + FP + Classification\ Errors} \times 100$$