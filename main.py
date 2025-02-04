import cv2
import mediapipe as mp
from deepface import DeepFace
from gtts import gTTS
import os

# Initialize Mediapipe
# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

# Open Camera
cap = cv2.VideoCapture(0)

# Default emotion
emotion = "unknown"  # Default emotion if not detected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    face_results = face_mesh.process(rgb_frame)

    # Detect hand landmarks
    hand_results = hands.process(rgb_frame)

    # Emotion Detection
    if face_results.multi_face_landmarks:
        cv2.imwrite("temp.jpg", frame)
        try:
            emotion = DeepFace.analyze("temp.jpg", actions=['emotion'])[0]['dominant_emotion']
        except Exception as e:
            print(f"Error during emotion detection: {e}")
            emotion = "unknown"  # Fallback in case of error

    # Gesture Analysis
    gesture = "No hand detected."
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            if thumb_tip.y < index_tip.y:
                gesture = "Thumbs up! Good job!"

    # Display results on frame
    text = f"Emotion: {emotion}, Gesture: {gesture}"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face & Hand Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate voice summary
summary_text = f"You are {emotion} with gesture: {gesture}."
tts = gTTS(summary_text, lang='en')
tts.save("summary.mp3")
os.system("start summary.mp3")  # For Windows, use 'open summary.mp3' on macOS
