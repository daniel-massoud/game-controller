import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import pyautogui
import time

# Safety — stops pyautogui if mouse hits corner
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# Load model
with open("gesture_model.pkl", "rb") as f:
    saved = pickle.load(f)
    gesture_model = saved["model"]
    encoder = saved["encoder"]

# Load MediaPipe
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Gesture to key mapping
GESTURE_KEYS = {
    "tilt_left":  "left",
    "tilt_right": "right",
    "point_up":   "up",
    "point_down": "down",
}

def normalize_landmarks(hand):
    wrist_x = hand[0].x
    wrist_y = hand[0].y
    wrist_z = hand[0].z
    coords = []
    for landmark in hand:
        coords.append([
            landmark.x - wrist_x,
            landmark.y - wrist_y,
            landmark.z - wrist_z
        ])
    scale = np.sqrt(
        coords[9][0]**2 +
        coords[9][1]**2 +
        coords[9][2]**2
    )
    if scale > 0:
        coords = [[c[0]/scale, c[1]/scale, c[2]/scale] for c in coords]
    flat = []
    for c in coords:
        flat.extend(c)
    return flat

# Gesture stabilization
gesture_buffer = []
BUFFER_SIZE = 5
last_gesture = None
last_key_time = 0
KEY_COOLDOWN = 0.3  # seconds between key presses

def get_stable_gesture(new_gesture):
    gesture_buffer.append(new_gesture)
    if len(gesture_buffer) > BUFFER_SIZE:
        gesture_buffer.pop(0)
    if len(gesture_buffer) == BUFFER_SIZE and len(set(gesture_buffer)) == 1:
        return gesture_buffer[0]
    return None

cap = cv2.VideoCapture(0)
print("Game Controller started.")
print("Click on your game window, then use gestures to play.")
print("Point up=jump | Point down=slide | Tilt left=left | Tilt right=right")
print("Press Q in this window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    h, w, _ = frame.shape
    detected_gesture = None
    stable_gesture = None

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw landmarks
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Predict
        normalized = normalize_landmarks(hand)
        detected_gesture = encoder.inverse_transform(
            gesture_model.predict([normalized])
        )[0]

        stable_gesture = get_stable_gesture(detected_gesture)

        # Send key press with cooldown
        if stable_gesture and stable_gesture in GESTURE_KEYS:
            now = time.time()
            if now - last_key_time > KEY_COOLDOWN:
                key = GESTURE_KEYS[stable_gesture]
                pyautogui.press(key)
                last_key_time = now
                last_gesture = stable_gesture

    # UI overlay
    gesture_colors = {
        "tilt_left":  (255, 128, 0),
        "tilt_right": (0, 128, 255),
        "point_up":   (0, 255, 0),
        "point_down": (0, 0, 255),
    }

    raw_text = detected_gesture if detected_gesture else "none"
    stable_text = stable_gesture if stable_gesture else "waiting..."
    color = gesture_colors.get(stable_gesture, (255, 255, 255))

    cv2.putText(frame, f"Detected: {raw_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"Action: {stable_text}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Key indicator boxes
    keys_display = [
        ("UP", "point_up",   (w//2 - 40, 20)),
        ("DOWN", "point_down", (w//2 - 40, 120)),
        ("LEFT", "tilt_left",  (20, 70)),
        ("RIGHT", "tilt_right", (w - 110, 70)),
    ]
    for label, gesture, pos in keys_display:
        active = stable_gesture == gesture
        box_color = gesture_colors.get(gesture, (100, 100, 100))
        alpha_color = box_color if active else (60, 60, 60)
        cv2.rectangle(frame, pos, (pos[0]+90, pos[1]+40), alpha_color, -1 if active else 2)
        cv2.putText(frame, label, (pos[0]+10, pos[1]+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Game Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()