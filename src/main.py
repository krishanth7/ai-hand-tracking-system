import cv2
import time
import pyautogui
import numpy as np
from tracker import HandTracker
from recognizer import GestureRecognizer
from utils import DrawingHelper, calculate_distance, map_range
from performance import LandmarkSmoother

# PyAutoGUI settings
pyautogui.FAILSAFE = False

class HandTrackingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker(max_hands=2, detection_con=0.7, track_con=0.7)
        self.smoother = LandmarkSmoother(alpha=0.6)
        self.p_time = 0
        self.c_time = 0
        
        # Screen dimensions for mouse control
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Virtual Drawing state
        self.draw_points = []
        self.is_drawing = False

    def run(self):
        print("Starting Hand Tracking System...")
        print("Press 'Q' to quit.")
        print("Press 'C' to clear drawing.")

        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            img = cv2.flip(img, 1) # Flip for mirroring
            img = self.tracker.find_hands(img)
            hands = self.tracker.get_position(img)
            
            gesture_display = "None"
            handedness_display = "None"

            if hands:
                for hand in hands:
                    # Smoothing
                    hand["lm_list"] = self.smoother.smooth(hand["label"], hand["lm_list"])
                    
                    fingers = self.tracker.fingers_up(hand)
                    gesture = GestureRecognizer.recognize(hand, fingers)
                    gesture_display = gesture
                    handedness_display = hand["label"]

                    # --- BONUS: Mouse Control ---
                    # Move cursor if pointing
                    if gesture == "Pointing":
                        index_tip = hand["lm_list"][8]
                        # Map frame coordinates to screen
                        screen_x = np.interp(index_tip[0], (100, self.frame_width - 100), (0, self.screen_width))
                        screen_y = np.interp(index_tip[1], (100, self.frame_height - 100), (0, self.screen_height))
                        pyautogui.moveTo(screen_x, screen_y)

                    # --- BONUS: Virtual Drawing ---
                    # Peace gesture to toggle drawing
                    if gesture == "Peace":
                        tip = hand["lm_list"][8]
                        self.draw_points.append((int(tip[0]), int(tip[1])))
                        self.is_drawing = True
                    else:
                        self.is_drawing = False

                    # --- BONUS: Volume Control (Pinch) ---
                    if gesture == "Pinch":
                        dist = calculate_distance(hand["lm_list"][4], hand["lm_list"][8])
                        # Map distance to volume percentage (simplified)
                        if dist < 50:
                            # pyautogui.press('volumedown') # Uncomment to actually control
                            cv2.putText(img, "Volume DOWN", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        elif dist > 150:
                            # pyautogui.press('volumeup') # Uncomment to actually control
                            cv2.putText(img, "Volume UP", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Draw virtual points
            for i in range(1, len(self.draw_points)):
                cv2.line(img, self.draw_points[i-1], self.draw_points[i], (255, 0, 255), 5)

            # FPS Calculation
            self.c_time = time.time()
            fps = 1 / (self.c_time - self.p_time)
            self.p_time = self.c_time

            # Overlay UI
            DrawingHelper.draw_header(img, fps, gesture_display, handedness_display)

            cv2.imshow("Production Hand Tracking System", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.draw_points = []

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandTrackingApp()
    app.run()
