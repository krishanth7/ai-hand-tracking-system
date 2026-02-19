import cv2
import mediapipe as mp
import numpy as np
import time
import os

# New MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    """
    Modern HandTracker using MediaPipe Tasks API.
    Compatible with newer Python versions (like 3.14) where legacy solutions might be missing.
    """
    def __init__(self, model_path='hand_landmarker.task', max_hands=2, detection_con=0.5, track_con=0.5):
        # Path to the .task model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        full_model_path = os.path.join(root_dir, model_path)
        
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found at {full_model_path}. Please ensure 'hand_landmarker.task' is in the project root.")

        base_options = python.BaseOptions(model_asset_path=full_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for simple integration
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        """Processes the image and finds hand landmarks."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Process the image
        self.results = self.detector.detect(mp_image)
        
        if draw and self.results.hand_landmarks:
            from utils import DrawingHelper
            DrawingHelper.draw_landmarks(img, self.results)
            
        return img

    def get_position(self, img):
        """
        Extracts landmark positions for all detected hands.
        Returns a list of dictionaries containing hand info.
        """
        hands_list = []
        if self.results and self.results.hand_landmarks:
            h, w, c = img.shape
            for i, hand_lms in enumerate(self.results.hand_landmarks):
                lm_list = []
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy, lm.z))
                
                # Get handedness
                # Mediapipe Tasks returns category_name as 'Left' or 'Right'
                label = self.results.handedness[i][0].category_name
                
                hands_list.append({
                    "id": i,
                    "lm_list": lm_list,
                    "label": label
                })
        return hands_list

    def fingers_up(self, hand):
        """Determines which fingers are open."""
        lm_list = hand["lm_list"]
        label = hand["label"]
        fingers = []
        
        # Thumb: Vertical and Horizontal logic
        # For 'Right' (as seen on screen), thumb is open if x is smaller than joint
        # Note: Handedness in Tasks API might be flipped compared to legacy
        if label == "Right":
            if lm_list[4][0] < lm_list[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lm_list[4][0] > lm_list[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # 4 Fingers: Vertical comparison
        tip_ids = [8, 12, 16, 20]
        for tip_id in tip_ids:
            if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
