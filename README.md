# Professional Hand Tracking System

A real-time, production-style Hand Tracking System built with Python, OpenCV, and MediaPipe. This project demonstrates advanced computer vision techniques for gesture recognition, landmark stabilization, and interactive control.

## 🚀 Features

- **Real-time Tracking**: Detects and tracks 21 hand landmarks for multiple hands.
- **Gesture Recognition**: Identifies Open Palm, Fist, Pointing, Peace, and Pinch gestures.
- **Landmark Stabilization**: Smooths landmark jitter using exponential moving average for a premium experience.
- **Clean UI Overlay**: Minimalist design showing FPS, active gesture, and handedness.
- **Bonus Interaction**:
  - **Mouse Control**: Control your cursor using your index finger (Pointing gesture).
  - **Virtual Drawing**: Draw in mid-air using the Peace gesture.
  - **Volume Feedback**: Visual cues for volume control during Pinch gestures.

## 🛠️ Tech Stack

- **Python**: Core logic.
- **OpenCV**: Video capture and image processing.
- **MediaPipe**: State-of-the-art hand tracking models.
- **PyAutoGUI**: System-level interaction for mouse and volume control.
- **NumPy**: Numerical transformations and coordinate mapping.

## 📂 Project Structure

```text
hand_tracking_system/
├── src/
│   ├── main.py           # Entry point & interaction logic
│   ├── tracker.py        # MediaPipe Hands wrapper
│   ├── recognizer.py     # Gesture analysis logic
│   ├── performance.py    # Stabilization and smoothing
│   └── utils.py          # Drawing and calculation helpers
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd hand_tracking_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python src/main.py
   ```

## 🎮 How to Use

- **Show Palm**: Displays "Open Palm".
- **Make a Fist**: Displays "Fist".
- **Point Index Finger**: Move the mouse cursor.
- **Peace Sign (Index + Middle)**: Start drawing on the screen.
- **Pinch (Thumb + Index)**: Triggers "Pinch" gesture (can be mapped to volume/zoom).
- **Press 'C'**: Clear the virtual drawing.
- **Press 'Q'**: Exit the application.

## 📝 Code Quality

This project follows **Object-Oriented Programming (OOP)** principles, with a focus on modularity, readability, and performance. Each component is isolated into its own module for easy maintenance and scalability.
