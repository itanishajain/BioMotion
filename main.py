import cv2
import mediapipe as mp
from deepface import DeepFace
from gtts import gTTS
import os
import threading
import time
import numpy as np
import pyttsx3

# Initialize Mediapipe solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create instances
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.8)
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Speech speed

# Open webcam
cap = cv2.VideoCapture(0)
running = True

# Function to detect emotion
def analyze_emotion(face_roi, person_id):
    """ Detect emotion for a specific face """
    global emotions
    try:
        small_frame = cv2.resize(face_roi, (300, 300))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        if analysis:
            emotions[person_id] = analysis[0]['dominant_emotion']
    except Exception as e:
        emotions[person_id] = "Neutral"

# Sign language mapping
sign_language_dict = {
    "thumbs_up": "Good Job!",
    "open_palm": "Hello!",
    "fist": "Stop!",
    "pointing": "Look at this!"
}

# Function to convert text to speech
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Tracking last spoken gesture
last_spoken = None
last_time = time.time()
recognized_text = "Waiting for sign..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_results = face_detection.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    emotions = {}  # Store emotions per person
    person_count = 0

    # Process detected faces
    if face_results.detections:
        for i, detection in enumerate(face_results.detections):
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, box_width, box_height = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                           int(bboxC.width * w), int(bboxC.height * h))
            
            face_roi = frame[y:y+box_height, x:x+box_width]  # Extract face region
            threading.Thread(target=analyze_emotion, args=(face_roi, i)).start()

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 0, 0), 2)
            person_count += 1

    # Process detected hands
    gesture_info = {}
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            wrist = hand_landmarks.landmark[0]

            detected_gesture = "unknown"
            if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
                detected_gesture = "thumbs_up"
            elif index_tip.y < wrist.y:
                detected_gesture = "open_palm"
            elif thumb_tip.x < index_tip.x and thumb_tip.x < middle_tip.x:
                detected_gesture = "pointing"
            elif all(lm.y > wrist.y for lm in hand_landmarks.landmark[4:9]):
                detected_gesture = "fist"

            # Store detected gesture
            gesture_info[i] = sign_language_dict.get(detected_gesture, "No gesture detected.")

            # Speak only if it's a new gesture
            if detected_gesture in sign_language_dict:
                recognized_text = sign_language_dict[detected_gesture]
                if detected_gesture != last_spoken or (time.time() - last_time > 2):
                    speak_text(recognized_text)
                    last_spoken = detected_gesture
                    last_time = time.time()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Process detected poses (posture tracking)
    posture_feedback = {}
    if pose_results.pose_landmarks:
        for i in range(person_count):
            landmarks = pose_results.pose_landmarks.landmark

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            hip_diff = abs(left_hip.y - right_hip.y)

            posture_color = (0, 255, 0)  # Default Green
            feedback = "Good posture."

            if shoulder_diff > 0.05:
                feedback = "Align your shoulders."
                posture_color = (0, 0, 255)
            elif hip_diff > 0.05:
                feedback = "Keep hips level."
                posture_color = (0, 0, 255)

            posture_feedback[i] = feedback

            # Draw skeleton for each person
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=posture_color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    # Display results on screen
    for i in range(person_count):
        text = f"Person {i+1}: Emotion: {emotions.get(i, 'Analyzing...')}, Gesture: {gesture_info.get(i, 'None')}, Posture: {posture_feedback.get(i, 'Analyzing...')}"
        cv2.putText(frame, text, (20, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display recognized sign language text
    cv2.putText(frame, recognized_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Multi-Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
