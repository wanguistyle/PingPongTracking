import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class BallTracker:
    def __init__(self, buffer_size=30, max_jump_dist=200, window_name="Smart Tracker"):
        self.pts = deque(maxlen=buffer_size)
        self.w_history = deque(maxlen=5) 
        self.window_name = window_name
        self.min_circularity = 0.5
        self.max_jump_dist = max_jump_dist
        self.last_center = None 
        self.last_w = 0
        self.velocity = (0, 0)
        self.missing_frames = 0
        self.MAX_COAST_FRAMES = 7 

    def setup_trackbars(self):
        def nothing(x): pass
        cv2.createTrackbar("Lower Hue", self.window_name, 5, 179, nothing)
        cv2.createTrackbar("Upper Hue", self.window_name, 25, 179, nothing)
        cv2.createTrackbar("Lower Sat", self.window_name, 130, 255, nothing)
        cv2.createTrackbar("Upper Sat", self.window_name, 255, 255, nothing)
        cv2.createTrackbar("Lower Val", self.window_name, 100, 255, nothing)
        cv2.createTrackbar("Upper Val", self.window_name, 255, 255, nothing)

    def process(self, frame):
        frame_h, frame_w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        l_h = cv2.getTrackbarPos("Lower Hue", self.window_name)
        u_h = cv2.getTrackbarPos("Upper Hue", self.window_name)
        l_s = cv2.getTrackbarPos("Lower Sat", self.window_name)
        u_s = cv2.getTrackbarPos("Upper Sat", self.window_name)
        l_v = cv2.getTrackbarPos("Lower Val", self.window_name)
        u_v = cv2.getTrackbarPos("Upper Val", self.window_name)

        mask = cv2.inRange(hsv, (l_h, l_s, l_v), (u_h, u_s, u_v))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        is_real = False 
        
        valid_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50: continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            
            # --- NOISE FILTER: Size Consistency ---
            if self.last_w > 0:
                if w > self.last_w * 2.5 or w < self.last_w * 0.4:
                    continue 

            if circularity > self.min_circularity and 0.5 < aspect < 1.5:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                if self.last_center is not None:
                    dist = np.linalg.norm(np.array((cX, cY)) - np.array(self.last_center))
                    if dist < self.max_jump_dist:
                        valid_candidates.append((c, (cX, cY), area, w))
                else:
                    valid_candidates.append((c, (cX, cY), area, w))

        if valid_candidates:
            c, center, _, raw_w = max(valid_candidates, key=lambda item: item[2])
            self.velocity = (center[0] - self.last_center[0], center[1] - self.last_center[1]) if self.last_center else (0,0)
            self.last_center, self.last_w, self.missing_frames = center, raw_w, 0
            is_real = True 
        else:
            # Prediction logic (Coasting)
            if self.last_center and self.missing_frames < self.MAX_COAST_FRAMES:
                self.missing_frames += 1
                center = (int(self.last_center[0] + self.velocity[0]), int(self.last_center[1] + self.velocity[1]))
                self.last_center = center
            else:
                self.last_center = None

        self.pts.appendleft(center)
        return frame, center, self.last_w, is_real

class PingPongUmpire:
    def __init__(self):
        self.history = []
        self.bounces_on_table = 0
        self.cooldown = 0

    def reset_rally(self):
        self.history.clear()
        self.bounces_on_table = 0
        self.cooldown = 0

    def update(self, pixel_x, pixel_y):
        if self.cooldown > 0: self.cooldown -= 1
        self.history.append((pixel_x, pixel_y))
        if len(self.history) > 12: self.history.pop(0)
        return self._analyze_precise()

    def _analyze_precise(self):
        if len(self.history) < 9 or self.cooldown > 0:
            return None

        xs = [pt[0] for pt in self.history]
        ys = [pt[1] for pt in self.history]
        mid = len(ys) // 2
        
        # 1. Slope Consistency Check (Filtering Noise)
        # Average speed of approach and departure
        slope_in = (ys[mid] - ys[mid-4]) / 4.0
        slope_out = (ys[mid+4] - ys[mid]) / 4.0
        
        # 2. Peak Detection (Is this the lowest point?)
        is_peak = ys[mid] == max(ys[mid-1:mid+2])
        
        # Threshold: Ball must be moving vertically at least 4 pixels/frame
        if is_peak and slope_in > 4 and slope_out < -4:
            # 3. Directional Reversal Logic
            v_in_x = xs[mid] - xs[mid-4]
            v_out_x = xs[mid+4] - xs[mid]
            
            # If x-direction flipped, product is negative
            direction_flipped = (v_in_x * v_out_x) < -15 
            
            self.cooldown = 20 
            
            if direction_flipped:
                self.bounces_on_table = 0
                return "TABLE -> PADDLE (REVERSAL)", (int(xs[mid]), int(ys[mid]))
            else:
                self.bounces_on_table += 1
                if self.bounces_on_table == 1:
                    return "BOUNCE 1 (TABLE)", (int(xs[mid]), int(ys[mid]))
                else:
                    return "DOUBLE BOUNCE (FAULT)", (int(xs[mid]), int(ys[mid]))
        return None

class PlayerDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, model_complexity=1)
        self.draw = mp.solutions.drawing_utils
        self.last_landmarks = None # Pour mémoriser la pose de la frame actuelle

    def process(self, frame):
        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.last_landmarks = res.pose_landmarks # On sauvegarde les points
        if res.pose_landmarks:
            self.draw.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return frame

    def classifier_coup(self):
        """Classifie le coup basé sur la position du poignet et du coude."""
        if not self.last_landmarks:
            return "Inconnu"
            
        # Index MediaPipe : 16 = Poignet droit, 14 = Coude droit
        poignet = self.last_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        coude = self.last_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        
        # En MediaPipe, les coordonnées x vont de 0.0 (gauche) à 1.0 (droite)
        if poignet.x < coude.x:
            return "COUP DROIT"
        else:
            return "REVERS"