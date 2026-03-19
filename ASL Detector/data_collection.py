import os 
import cv2
import numpy as np
import mediapipe as mp
import csv

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence = 0.7, max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

CSV_FILE = "data_collection.csv"
collecting = False
count = 0

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "a+", newline="") as f:
        headers = []
        writer = csv.writer(f)
        for i in range(21):
            headers += [f"x{i}", f"y{i}"]
        headers.append("label")
        writer.writerow(headers)

current_letter = "A"

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    landmarks = []
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = img.shape 
            #extracting 21 hand landmarks 
            for lm in handLms.landmark:
                landmarks += [lm.x, lm.y]
        
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if collecting and len(landmarks) == 42:
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks + [current_letter])
                count += 1

        # UI
    cv2.putText(img, f"Letter : {current_letter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(img, f"Samples: {count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img, "HOLD SPACE to collect  |  A-Z to switch letter  |  Q to quit", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if collecting:
        cv2.putText(img, "COLLECTING...", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) 
    cv2.imshow("Data Collection", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('2'):
        break
    elif key == 32: # csv file will be appended only when space bar is clicked
        collecting = True
    else:
        collecting = False

    if 65 <= key <= 90 or 97<= key <= 122:
        current_letter = chr(key).upper()
        count = 0

cap.release()
cv2.destroyAllWindows()

            




