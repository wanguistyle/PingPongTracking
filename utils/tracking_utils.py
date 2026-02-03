import cv2
import numpy as np
from collections import deque

class TableDetector:
    def __init__(self, reference_image_path, min_match_count=10):
        self.min_match_count = min_match_count
        self.sift = cv2.SIFT_create()
        
        self.img_base = cv2.imread(reference_image_path, 0)
        if self.img_base is None:
            raise ValueError(f"Could not load image at {reference_image_path}")
            
        self.kp_base, self.des_base = self.sift.detectAndCompute(self.img_base, None)
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def process(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate keypoints for the current frame
        kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)

        if des_frame is None or len(des_frame) < 2:
            return frame

        matches = self.matcher.knnMatch(self.des_base, des_frame, k=2)

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) > self.min_match_count:
            src_pts = np.float32([self.kp_base[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            
            # FIXED: Used 'kp_frame' directly instead of 'self.kp_frame'
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = self.img_base.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "Table Locked", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame

class BallTracker:
    def __init__(self, buffer_size=20, window_name="Smart Tracker"):
        self.pts = deque(maxlen=buffer_size)
        self.window_name = window_name
        self.min_circularity = 0.5
        
    def setup_trackbars(self):
        def nothing(x): pass
        cv2.createTrackbar("Lower Hue", self.window_name, 5, 179, nothing)
        cv2.createTrackbar("Upper Hue", self.window_name, 25, 179, nothing)
        cv2.createTrackbar("Lower Sat", self.window_name, 130, 255, nothing)
        cv2.createTrackbar("Upper Sat", self.window_name, 255, 255, nothing)
        cv2.createTrackbar("Lower Val", self.window_name, 100, 255, nothing)
        cv2.createTrackbar("Upper Val", self.window_name, 255, 255, nothing)

    def process(self, frame):
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

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(c, True)
            area = cv2.contourArea(c)

            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)

                if area > 300 and circularity > self.min_circularity and 0.6 < aspect < 1.4:
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)

        self.pts.appendleft(center)
        
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None: continue
            thickness = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

        return frame