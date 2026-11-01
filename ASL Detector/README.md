# ASL Alphabet Recognition

Real-time American Sign Language (ASL) alphabet recognition from a webcam, using MediaPipe hand landmarks and a Random Forest classifier.

## How It Works
1. **Hand tracking:** MediaPipe Hands finds one hand per frame and returns 21 landmarks.
2. **Features:** the x and y coordinates of the 21 landmarks give 42 features per frame. Using landmarks instead of raw pixels lets the model learn hand shape, independent of background and lighting.
3. **Classifier:** a Random Forest (100 trees) predicts the letter.
4. **Live prediction:** the trained model labels each webcam frame and shows the predicted letter with its confidence.

## Dataset
I collected a custom dataset with `data_collection.py`:
- 8,512 labeled samples
- All 26 letters, A–Z
- 301–422 samples per letter

## Results
| Evaluation | Accuracy |
|---|---|
| 80/20 train-test split | 99.7% |
| 5-fold stratified cross-validation | 99.7% (± 0.1%) |

### Limitations
- **Optimistic test set.** Samples are consecutive frames of one signer, so test frames closely resemble training frames. Testing on a new session or a different signer would give a more realistic estimate.
- **Motion letters.** J and Z are signed with movement, but the model classifies single static frames.
- **One hand, 2D only.** The model uses one hand and ignores landmark depth (z).

## Files
| File | Purpose |
|---|---|
| `data_collection.py` | Webcam tool for recording labeled landmark samples to `data_collection.csv` |
| `data_collection.csv` | The collected dataset (42 features + label) |
| `model.py` | Trains the Random Forest, prints a classification report, plots a confusion matrix, saves `sign_model.pkl` |
| `main.py` | Runs live prediction from the webcam |

## Setup
```bash
pip install opencv-python "mediapipe<0.10.15" scikit-learn pandas matplotlib seaborn
```
The scripts use MediaPipe's `mp.solutions.hands` API, which newer MediaPipe releases may no longer include.

## Usage
1. **Collect data (optional):** `python data_collection.py`
   - Press a letter key (A–Z) to choose the letter to record.
   - Hold **Space** to record samples.
   - Press **2** to quit.
2. **Train:** `python model.py`. Update `CSV_FILE` at the top of the file to point to your `data_collection.csv`. This creates `sign_model.pkl`.
3. **Run live:** `python main.py`. Press **Q** to quit.

## Tech Stack
Python, OpenCV, MediaPipe, scikit-learn, pandas, matplotlib, seaborn
