# Hand Gesture Game Controller

Control any arrow-key game using hand gestures and your webcam.
No keyboard needed, just your hand.

---

## Gestures

| Gesture | Action |
|---------|--------|
| Point up | Jump (↑) |
| Point down | Slide (↓) |
| Tilt left | Move left (←) |
| Tilt right | Move right (→) |
| Fist | Neutral - do nothing |

## How it works

1. MediaPipe detects your hand and extracts 21 landmarks per frame
2. Landmarks are normalized to be scale and position invariant
3. A neural network classifies the gesture in real time
4. PyAutoGUI sends the corresponding keypress to whatever game is active

## The ML pipeline

- Collected 4000 samples across 5 gesture classes
- Applied wrist-relative normalization for scale invariance
- Trained a 3-layer neural network (256 → 128 → 64)
- Achieved 99.6% accuracy on held-out test set
- Runs at real-time speed on CPU

## Works with any game that uses arrow keys

- Temple Run
- Subway Surfers
- Any browser or desktop game with arrow key controls

## Setup

    git clone https://github.com/daniel-massoud/game-controller
    cd game-controller
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Download the MediaPipe hand landmarker model:

    python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')"

Collect your own gesture data:

    python collect_data.py

Train the model on your hand:

    python train_model.py

Run the controller:

    python controller.py

Then click on your game window and start playing.

## Tech stack

- Python
- MediaPipe -> hand landmark detection
- OpenCV -> real-time video processing
- scikit-learn -> neural network classifier
- PyAutoGUI -> keyboard simulation
- NumPy -> landmark normalization
