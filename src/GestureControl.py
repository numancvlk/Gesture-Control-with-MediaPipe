#LIBRARIES
import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown = 0.8
current_gesture_display = "Ready"

def is_fist(landmarks):
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)] 
    closed_count = 0
    for tip, pip in fingers:
        if landmarks[tip].y > landmarks[pip].y:  
            closed_count += 1
    return closed_count == 4

def to_px(lm, w, h):
    """Convert normalized landmarks to pixel coordinates."""
    return int(lm.x * w), int(lm.y * h)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            thumb_x, thumb_y = to_px(landmarks[4], w, h)
            wrist_x, wrist_y = to_px(landmarks[0], w, h)

            current_time = time.time()

            if current_time - last_action_time > cooldown:
                # --- FIST ---
                if is_fist(landmarks):
                    current_gesture_display = "PLAY / PAUSE"
                    pyautogui.press("playpause")
                    last_action_time = current_time

                # --- THUMB DIRECTIONS ---
                elif thumb_y < wrist_y - 60:
                    current_gesture_display = "VOLUME UP"
                    pyautogui.press("volumeup")
                    last_action_time = current_time
                elif thumb_y > wrist_y + 60:
                    current_gesture_display = "VOLUME DOWN"
                    pyautogui.press("volumedown")
                    last_action_time = current_time
                elif thumb_x < wrist_x - 60:
                    current_gesture_display = "PREVIOUS TRACK"
                    pyautogui.hotkey("prevtrack")
                    last_action_time = current_time
                elif thumb_x > wrist_x + 60:
                    current_gesture_display = "NEXT TRACK"
                    pyautogui.hotkey("nexttrack")
                    last_action_time = current_time
                else:
                    current_gesture_display = "Idle"
    else:
        current_gesture_display = "No Hand Detected"

    # --- HUD ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 80), (50, 50, 50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, "Hand Gesture Control", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Action: {current_gesture_display}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Control with HUD", frame)

    if cv2.waitKey(5) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
