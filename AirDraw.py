import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

save_dir = os.path.expanduser("~/Downloads/cizimler")
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_points = {}
drawing = True
last_save_time = 0
status_text = ""

def count_fingers(hand_landmarks, handedness_str):
    fingers = []

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if handedness_str == 'Right':
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    status_text = "Eller algilanmadi."
    hand_infos = []

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_tip = int(hand_landmarks.landmark[8].x * w)
            y_tip = int(hand_landmarks.landmark[8].y * h)

            finger_count = count_fingers(hand_landmarks, label)

            hand_infos.append({
                "label": label,
                "finger_count": finger_count,
                "x_tip": x_tip,
                "y_tip": y_tip,
                "landmarks": hand_landmarks
            })

            if label not in prev_points:
                prev_points[label] = (0, 0)

        # Yumruk temizleme
        if any(hand["finger_count"] == 0 for hand in hand_infos):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            prev_points = {}
            status_text = "Yumruk: Temizlendi"
            drawing = True
        else:
            drawing = True
            for hand in hand_infos:
                label = hand["label"]
                finger_count = hand["finger_count"]
                x_tip = hand["x_tip"]
                y_tip = hand["y_tip"]

                if finger_count == 1:
                    prev_x, prev_y = prev_points[label]
                    if prev_x == 0 and prev_y == 0:
                        prev_points[label] = (x_tip, y_tip)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (255, 0, 0), 5)
                        prev_points[label] = (x_tip, y_tip)
                else:
                    prev_points[label] = (0, 0)

            # Kaydetme hareketi: Sol el okay işareti
            for hand in hand_infos:
                if hand["label"] == "Left":
                    lm = hand["landmarks"].landmark
                    dist = distance(lm[4], lm[8])  # Başparmak ucu ve işaret parmağı ucu
        
                   # Okay işareti: mesafe küçük ve 3 parmak açık
                    okay = (dist < 0.05 and hand["finger_count"] == 3)

                  # Thumbs up işareti: başparmak yukarı açık, diğer parmaklar kapalı
                    thumb_up = (lm[4].y < lm[3].y and  # başparmak ucu yukarıda
                                lm[8].y > lm[6].y and  # işaret parmağı kapalı
                                lm[12].y > lm[10].y and # orta parmak kapalı
                                lm[16].y > lm[14].y and # yüzük parmağı kapalı
                                lm[20].y > lm[18].y)    # serçe parmak kapalı

                    if okay or thumb_up:
                        current_time = time.time()
                        if current_time - last_save_time > 2:
                            filename = f"cizim_{int(current_time)}.png"
                            filepath = os.path.join(save_dir, filename)
                            cv2.imwrite(filepath, canvas)
                            last_save_time = current_time
                            status_text = f"Kaydedildi: {filename} " # 👌veya👍
                            drawing = False
                     
                        break


    else:
        prev_points = {}
        drawing = True

    overlay = frame.copy()
    cv2.rectangle(overlay, (5,5), (w//2, 90), (50,50,50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, "Kontroller:", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Yumruk: Temizle", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, "Sol el ok: Kaydet", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(frame, status_text, (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Air Draw - Havaya Ciz", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()