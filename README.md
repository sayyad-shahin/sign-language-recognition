# 🤟 Sign Language Recognition (Real-Time, Webcam)

A real-time hand-sign recognition system. **MediaPipe** extracts 21 hand landmarks from the webcam, the landmarks are normalized into a 63-number feature vector, and a **scikit-learn** classifier predicts the sign. Hold a sign steady and it is typed into a sentence (with optional text-to-speech).

Because it classifies *landmarks* rather than raw pixels, it is fast (CPU only, real time), small, and robust to background, lighting, skin tone, hand position and distance from the camera.

## Features
- Live webcam recognition with confidence score and bounding box
- Prediction smoothing (majority vote) + hold-to-type sentence builder
- Train on **your own webcam data** (`collect_data.py`) or on an **image dataset** such as Kaggle ASL Alphabet (`prepare_from_images.py`)
- Compares RandomForest, Logistic Regression and MLP and keeps the best
- Works with either hand (left hand is mirrored automatically)
- Works for any signs: letters, numbers or custom words/gestures
- Single-image prediction script, unit tests, confusion matrix + metrics

## Project structure
```
sign-language-recognition/
├── features.py              # MediaPipe hand detector + landmark normalization
├── collect_data.py          # record training data from webcam
├── prepare_from_images.py   # build training data from an image dataset
├── train.py                 # train, compare, save best model
├── classifier.py            # load model + predict
├── recognize.py             # real-time recognition app
├── predict_image.py         # predict from one image
├── tests/                   # pytest tests
├── data/                    # landmarks.csv goes here
└── models/                  # trained model + metrics (generated)
```

## Setup
Python 3.9 - 3.12 (MediaPipe does not yet support newer versions with this API).
```bash
git clone https://github.com/sayyad-shahin/sign-language-recognition.git
cd sign-language-recognition
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Step 1 - Get training data (pick one)
**A) Record with your webcam (recommended, ~10 min for A-Z):**
```bash
python collect_data.py                       # letters A-Z, 200 samples each
python collect_data.py --labels HELLO YES NO THANKS --samples 300   # custom signs
```
Press `SPACE` to record a sign, move/rotate your hand slightly while recording, `N` to skip, `Q` to quit. Data is appended to `data/landmarks.csv`, so you can continue later.

**B) Use an image dataset** (folders named by label, e.g. Kaggle *ASL Alphabet*):
```bash
python prepare_from_images.py --dataset dataset --max-per-class 300
```

### Step 2 - Train
```bash
python train.py
```
Prints accuracy per model + classification report, and saves `models/sign_model.joblib`, `models/metrics.json` and `models/confusion_matrix.png`.

### Step 3 - Run
```bash
python recognize.py
```
| Key | Action |
|---|---|
| hold a sign ~1 s | types the letter |
| `SPACE` | add space |
| `BACKSPACE` | delete last char |
| `C` | clear |
| `S` | speak sentence (needs `pip install pyttsx3`) |
| `Q` | quit |

Options: `--camera 1`, `--threshold 0.8` (stricter), `--hold 15` (faster typing).

```bash
python predict_image.py photo.jpg     # single image
pytest                                # run tests
```

## How it works
1. **Detect** - MediaPipe Hands finds 21 3D landmarks per frame.
2. **Normalize** - wrist moved to origin, scaled by hand size, left hand mirrored → position/scale/hand invariant 63-D feature vector.
3. **Classify** - RandomForest / Logistic Regression / MLP trained on the vectors; best test accuracy wins.
4. **Stabilize** - majority vote over the last 10 frames + confidence threshold + hold timer avoids flicker and accidental letters.

## Results
Fill in after training on your own data (see `models/metrics.json`):

| Model | Test accuracy |
|---|---|
| RandomForest | _fill in_ |
| Logistic Regression | _fill in_ |
| MLP | _fill in_ |

Add a screenshot/GIF of the live app here: `![demo](docs/demo.gif)`

## Limitations
- Recognizes **static** hand shapes (e.g. fingerspelling). Motion signs such as J and Z, or full sentences in ASL/ISL, need sequence models (LSTM/Transformer over landmark sequences).
- Accuracy depends on the training data; record with varied hand angles/distances, ideally with several people.
- Similar shapes (e.g. A/S/T, M/N) can be confused.
- Single hand only.

## Future improvements
Two-hand support · LSTM/Transformer for dynamic signs · Indian Sign Language dataset · Streamlit/web version · Dockerfile · TFLite export for mobile

## Author
**Shahin Sayyad** · [LinkedIn](https://linkedin.com/in/shahin-sayyad1) · [GitHub](https://github.com/sayyad-shahin)

## License
MIT
