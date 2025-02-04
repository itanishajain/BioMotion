import cv2
import mediapipe as mp
from deepface import DeepFace
from gtts import gTTS
import os
import threading
import time

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

# Open Camera
cap = cv2.VideoCapture(0)

# Variables for storing results
emotion = "Unknown"
gesture = "No gesture detected."
last_emotion_time = 0
last_speech_time = 0
running = True  # Control flag for speaking
prev_speech_text = ""

# Function to detect emotion in a separate thread
def analyze_emotion(frame):
    global emotion
    try:
        cv2.imwrite("temp.jpg", frame)
        analysis = DeepFace.analyze("temp.jpg", actions=['emotion'])
        emotion = analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        emotion = "Unknown"

# Function to handle speech synthesis
def speak(text):
    global prev_speech_text
    if text != prev_speech_text:  # Speak only if the text has changed
        tts = gTTS(text, lang='en')
        tts.save("temp_voice.mp3")
        os.system("start temp_voice.mp3")  # Windows: 'start', Mac: 'afplay'
        prev_speech_text = text

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

    # Run emotion detection every 5 seconds
    current_time = time.time()
    if current_time - last_emotion_time > 5 and face_results.multi_face_landmarks:
        last_emotion_time = current_time
        threading.Thread(target=analyze_emotion, args=(frame,)).start()

    # Gesture Analysis
    gesture = "No gesture detected."
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            if thumb_tip.y < index_tip.y:
                gesture = "Thumbs up! Good job!"
            else:
                gesture = "Hand detected, no thumbs up."

    # Display results on frame
    text = f"Emotion: {emotion}, Gesture: {gesture}"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Speak only when the webcam is running & every 3 seconds
    if running and (current_time - last_speech_time > 3):
        last_speech_time = current_time
        threading.Thread(target=speak, args=(text,)).start()

    # Show frame
    cv2.imshow("Face & Hand Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False  # Stop speech
        break

cap.release()
cv2.destroyAllWindows()