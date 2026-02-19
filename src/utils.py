import cv2
import numpy as np
import math

class DrawingHelper:
    """
    Helper class for drawing overlays on the video feed.
    """
    
    # Professional Color Palette (BGR)
    COLORS = {
        'PRIMARY': (255, 120, 0),    # Cyan/Blue
        'SECONDARY': (0, 255, 127),  # Sea Green
        'ACCENT': (0, 0, 255),       # Red
        'TEXT': (255, 255, 255),      # White
        'CONNECTION': (200, 200, 200) # Light Gray
    }

    # Landmark connections (indexes)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm/Knuckles
        (5, 9), (9, 13), (13, 17)
    ]

    @staticmethod
    def draw_landmarks(image, results):
        """Manually draws hand landmarks and connections."""
        if not results.hand_landmarks:
            return

        h, w, _ = image.shape
        for hand_landmarks in results.hand_landmarks:
            # 1. Draw Connections
            for connection in DrawingHelper.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                p1 = hand_landmarks[start_idx]
                p2 = hand_landmarks[end_idx]
                
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                
                cv2.line(image, pt1, pt2, DrawingHelper.COLORS['CONNECTION'], 2)

            # 2. Draw Landmarks
            for i, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Highlight tips
                if i in [4, 8, 12, 16, 20]:
                    color = DrawingHelper.COLORS['SECONDARY']
                    radius = 6
                else:
                    color = DrawingHelper.COLORS['PRIMARY']
                    radius = 4
                
                cv2.circle(image, (cx, cy), radius, color, -1)
                cv2.circle(image, (cx, cy), radius + 1, (255, 255, 255), 1)

    @staticmethod
    def draw_header(image, fps, gesture_name, handedness):
        """Draws a clean UI header with FPS and status."""
        h, w, _ = image.shape
        
        # Transparent background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # FPS Display
        cv2.putText(image, f"FPS: {int(fps)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Gesture Display
        cv2.putText(image, f"Gesture: {gesture_name}", (w // 2 - 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Handedness
        cv2.putText(image, f"Hand: {handedness}", (w - 150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def map_range(x, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another."""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
