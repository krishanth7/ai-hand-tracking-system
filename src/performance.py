import numpy as np

class LandmarkSmoother:
    """
    Implements a simple One Euro Filter or Moving Average to reduce jitter 
    in landmark positions.
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # Smoothing factor (0 to 1)
        self.previous_landmarks = {}

    def smooth(self, hand_id, landmarks):
        """
        Applies exponential moving average to landmarks.
        hand_id: Unique ID for the hand to maintain temporal consistency if possible.
        landmarks: List of (x, y, z) tuples.
        """
        if hand_id not in self.previous_landmarks:
            self.previous_landmarks[hand_id] = landmarks
            return landmarks

        smoothed_landmarks = []
        prev = self.previous_landmarks[hand_id]
        
        for p, s in zip(prev, landmarks):
            # SMA: smoothed = alpha * current + (1 - alpha) * previous
            sx = self.alpha * s[0] + (1 - self.alpha) * p[0]
            sy = self.alpha * s[1] + (1 - self.alpha) * p[1]
            sz = self.alpha * s[2] + (1 - self.alpha) * p[2]
            smoothed_landmarks.append((sx, sy, sz))
            
        self.previous_landmarks[hand_id] = smoothed_landmarks
        return smoothed_landmarks
