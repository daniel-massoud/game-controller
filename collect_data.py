import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os

model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

GESTURES = ["neutral"]

data_file = "gesture_data.csv"

if not os.path.exists(data_file):
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(63)]
        header.append("label")
        writer.writerow(header)

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

current_gesture_idx = 0
collecting = False
sample_count = 0
samples_per_gesture = 800

cap = cv2.VideoCapture(0)
print("Data collection started.")
print(f"Collecting gesture: {GESTURES[current_gesture_idx]}")
print("SPACE=start/stop  N=next gesture  Q=quit")
print("Vary distance, angle, and position on screen")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    current_gesture = GESTURES[current_gesture_idx]
    status = "COLLECTING" if collecting else "PAUSED"
    color = (0, 255, 0) if collecting else (0, 165, 255)

    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Samples: {sample_count}/{samples_per_gesture}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Vary: close/far/left/right/tilted", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "SPACE=collect  N=next  Q=quit", (10, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if result.hand_landmarks:
        h, w, _ = frame.shape
        hand = result.hand_landmarks[0]

        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if collecting and sample_count < samples_per_gesture:
            normalized = normalize_landmarks(hand)
            normalized.append(current_gesture)
            with open(data_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(normalized)
            sample_count += 1

        if sample_count >= samples_per_gesture:
            collecting = False
            cv2.putText(frame, "DONE! Press N for next", (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if sample_count < samples_per_gesture:
            collecting = not collecting
    elif key == ord('n'):
        if current_gesture_idx < len(GESTURES) - 1:
            current_gesture_idx += 1
            sample_count = 0
            collecting = False
            print(f"Now collecting: {GESTURES[current_gesture_idx]}")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")