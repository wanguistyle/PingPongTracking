import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class TablePnPEstimator:
    def __init__(self, frame_width, frame_height):
        self.obj_pts = np.float32([[0, 1370, 0], [0, 0, 0], [1525, 0, 0], [1525, 1370, 0]])
        focal_length = frame_width * 0.8
        self.cam_matrix = np.array([[focal_length, 0, frame_width / 2], [0, focal_length, frame_height / 2], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((4, 1))
        self.rvec = None
        self.tvec = None

    def update_camera_pose(self, image_pts):
        if image_pts is None or len(image_pts) != 4: return False
        success, self.rvec, self.tvec = cv2.solvePnP(self.obj_pts, np.float32(image_pts), self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        return success

    def project_ball_to_table_plane(self, ball_x_2d, ball_y_2d):
        if self.rvec is None or self.tvec is None: return None
        R, _ = cv2.Rodrigues(self.rvec)
        uv_point = np.array([[ball_x_2d], [ball_y_2d], [1.0]])
        K_inv = np.linalg.inv(self.cam_matrix)
        R_inv = np.linalg.inv(R)
        ray_dir = R_inv.dot(K_inv.dot(uv_point))
        cam_pos = -R_inv.dot(self.tvec)
        if ray_dir[2, 0] == 0: return None
        t = -cam_pos[2, 0] / ray_dir[2, 0]
        world_pt = cam_pos + t * ray_dir
        return (float(world_pt[0, 0]), float(world_pt[1, 0]))

class PingPongUmpire:
    def __init__(self):
        self.TABLE_WIDTH = 1525.0
        self.TABLE_LENGTH = 1370.0 
        self.history_2d_y = []
        self.history_table_coords = []

    def update(self, pixel_y, table_x, table_y):
        self.history_2d_y.append(pixel_y)
        self.history_table_coords.append((table_x, table_y))
        if len(self.history_2d_y) > 10:
            self.history_2d_y.pop(0)
            self.history_table_coords.pop(0)
        return self._detect_bounce()

    def _detect_bounce(self):
        if len(self.history_2d_y) < 3: return None
        
        y_prev2 = self.history_2d_y[-3]
        y_prev1 = self.history_2d_y[-2]
        y_current = self.history_2d_y[-1]
        
        # In video, Y increases as it goes DOWN the screen.
        # A bounce is when the ball goes down (y increases), then up (y decreases).
        if y_prev1 > y_prev2 and y_prev1 > y_current:
            bounce_x, bounce_y = self.history_table_coords[-2]
            return self._analyze_bounce(bounce_x, bounce_y)
        return None

    def _analyze_bounce(self, x, y):
        # Adding a 30mm grace period for balls that hit the very edge of the white line
        if -30 <= x <= self.TABLE_WIDTH + 30 and -30 <= y <= self.TABLE_LENGTH + 30:
            return "IN!"
        else:
            return "OUT!"

class CourtVisualizer:
    def __init__(self, scale=0.2):
        self.scale = scale
        self.width = int(1525 * scale)
        self.length = int(1370 * scale) 
        
    def draw(self, table_coords=None, text=""):
        minimap = np.zeros((self.length + 100, self.width + 100, 3), dtype=np.uint8)
        offset_x, offset_y = 50, 50
        
        cv2.rectangle(minimap, (offset_x, offset_y), (offset_x + self.width, offset_y + self.length), (150, 50, 50), -1)
        cv2.rectangle(minimap, (offset_x, offset_y), (offset_x + self.width, offset_y + self.length), (255, 255, 255), 2)
        cv2.line(minimap, (offset_x, offset_y), (offset_x + self.width, offset_y), (200, 200, 200), 4)

        if table_coords:
            bx, by = table_coords
            map_x = int(bx * self.scale) + offset_x
            map_y = offset_y + (self.length - int(by * self.scale))
            
            if 0 <= map_y <= minimap.shape[0] and 0 <= map_x <= minimap.shape[1]:
                cv2.circle(minimap, (map_x, map_y), 8, (0, 165, 255), -1)
                cv2.circle(minimap, (map_x, map_y), 8, (255, 255, 255), 1)

        if text:
            color = (0, 255, 0) if "IN" in text else (0, 0, 255)
            cv2.putText(minimap, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        return minimap
    
class BallTracker:
    def __init__(self, buffer_size=20, max_jump_dist=200, window_name="Smart Tracker"):
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
        current_w = 0
        
        valid_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50: continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            
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

        if len(valid_candidates) > 0:
            c, center, _, raw_w = max(valid_candidates, key=lambda item: item[2])
            x, y, w, h = cv2.boundingRect(c)
            
            if self.last_center is not None:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                self.velocity = (dx, dy)
            else:
                self.velocity = (0, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            
            self.last_center = center
            self.last_w = raw_w
            self.missing_frames = 0
            
            self.w_history.append(raw_w)
            current_w = np.mean(self.w_history)
            
        else:
            if self.last_center is not None and self.missing_frames < self.MAX_COAST_FRAMES:
                self.missing_frames += 1
                pred_x = self.last_center[0] + self.velocity[0]
                pred_y = self.last_center[1] + self.velocity[1]
                
                if 0 <= pred_x <= frame_w and 0 <= pred_y <= frame_h:
                    center = (int(pred_x), int(pred_y))
                    current_w = self.last_w
                    
                    half_w = int(current_w / 2)
                    cv2.rectangle(frame, (center[0] - half_w, center[1] - half_w), 
                                        (center[0] + half_w, center[1] + half_w), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 165, 255), -1)
                    
                    self.last_center = center
                else:
                    self.last_center = None
                    self.velocity = (0, 0)
                    self.missing_frames = 0
            else:
                self.last_center = None
                self.velocity = (0, 0)
                self.missing_frames = 0

        self.pts.appendleft(center)
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None: continue
            thickness = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

        return frame, center, current_w

class PlayerDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    def process(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        return frame