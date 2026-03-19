import cv2
import pickle
import mediapipe as mp
import numpy as np

with open("sign_model.pkl", "rb") as f:
    rf = pickle.load(f)

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence = 0.7, max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    landmarks = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for lm in handLms.landmark:
                landmarks += [lm.x, lm.y]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if len(landmarks) == 42:
            prediction = rf.predict([landmarks])[0]
            confidence = int(max(rf.predict_proba([landmarks])[0]) * 100)

            cv2.putText(img, f"{prediction}", (10,80), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0),5)
            cv2.putText(img, f"{confidence}", (10,140), cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0),5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Sign Language Predictor",img)

cap.release()
cv2.destroyAllWindows()


