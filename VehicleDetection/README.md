# Vehicle Detection and Counting

Detects, tracks and counts vehicles in traffic video with YOLOv8 and the SORT tracker.

## How It Works
1. **Region of interest:** `car_mask.png` blacks out everything except the road, so only relevant vehicles are detected.
2. **Detection:** YOLOv8 nano (`yolov8n.pt`) detects objects in each frame. Only cars, buses, trucks and motorbikes with confidence above 0.3 are kept.
3. **Tracking:** SORT (`sort.py`) assigns each vehicle a persistent ID across frames (max age 20 frames, 3 hits to confirm, IoU threshold 0.3).
4. **Counting:** a vehicle is counted when its center crosses a virtual line. Each tracker ID is counted only once, which prevents double counting.

The output window shows each tracked vehicle's ID, the counting line, the running count and the processing frame rate.

## Files
| File | Purpose |
|---|---|
| `main.py` | Detection, tracking and counting pipeline |
| `sort.py` | SORT multi-object tracker |
| `cars.mp4` | Sample traffic video |
| `car_mask.png` | Region-of-interest mask for the sample video |

## Setup
```bash
pip install ultralytics opencv-python cvzone numpy filterpy scikit-image scipy matplotlib
```
The YOLOv8 weights (`yolov8n.pt`) download automatically on first run.

## Usage
1. In `main.py`, update the paths in `cv2.VideoCapture(...)` and `cv2.imread(...)` to point to `cars.mp4` and `car_mask.png`.
2. Run `python main.py`.
3. Press any key to advance one frame, or **Q** to quit. The script waits for a key on every frame.

## Possible Improvements
- Measure counting accuracy against a hand-labeled count for the video.
- Count each vehicle class separately and by direction of travel.
- Play the video continuously instead of frame by frame.

## Tech Stack
Python, Ultralytics YOLOv8, OpenCV, cvzone, SORT, NumPy
