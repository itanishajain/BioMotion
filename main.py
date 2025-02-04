import cv2
import mediapipe as mp
from deepface import DeepFace
from gtts import gTTS
import os
import threading
import time
import numpy as np

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8)
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Open Camera
cap = cv2.VideoCapture(0)

# Variables for storing results
emotion = "Neutral"
gesture = "No gesture detected."
feedback = "Adjust posture for better recognition."
summary = ""
last_emotion_time = 0
last_speech_time = 0
running = True  # Control flag for speaking
prev_speech_text = ""

# Function to detect emotion in a separate thread
def analyze_emotion(frame):
    global emotion, feedback, summary
    try:
        small_frame = cv2.resize(frame, (300, 300))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        if analysis:
            emotion = analysis[0]['dominant_emotion']
            feedback = f"Detected emotion: {emotion}. Maintain a confident expression."
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        emotion = "Neutral"
        feedback = "Face not detected clearly. Adjust lighting."

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

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    current_time = time.time()
    if current_time - last_emotion_time > 5 and face_results.multi_face_landmarks:
        last_emotion_time = current_time
        threading.Thread(target=analyze_emotion, args=(frame.copy(),)).start()

    gesture = "No gesture detected."
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            wrist = hand_landmarks.landmark[0]

            if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
                gesture = "Thumbs up! Well done."
            elif index_tip.y < wrist.y:
                gesture = "Open palm detected."
            elif thumb_tip.x < index_tip.x and thumb_tip.x < middle_tip.x:
                gesture = "Pointing gesture detected."
            else:
                gesture = "Hand detected, no recognized gesture."
    
    # Detect posture issues
    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        hip_diff = abs(left_hip.y - right_hip.y)

        if shoulder_diff > 0.05:
            feedback = "Adjust your shoulders to stand upright."
        elif hip_diff > 0.05:
            feedback = "Keep your hips aligned for better posture."

    # Attire and Face Analysis
    if face_results.multi_face_landmarks:
        feedback = "Face detected. "
        if emotion in ["angry", "sad"]:
            feedback += "Try smiling for a more positive expression."
        elif emotion in ["happy", "surprise"]:
            feedback += "Your facial expression looks great!"
        else:
            feedback += "Maintain a confident expression."
        feedback += " Fix your hair if it's messy for a neat look."
    else:
        feedback = "Face not fully detected, adjust position."
    
    summary = f"{gesture}. {feedback}"
    text = f"Emotion: {emotion}, Gesture: {gesture}, Feedback: {feedback}"
    cv2.putText(frame, summary, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if running and (current_time - last_speech_time > 4):
        last_speech_time = current_time
        threading.Thread(target=speak, args=(feedback,)).start()

    cv2.imshow("BioMotion - Virtual Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()