import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from cvfpscalc import CvFpsCalc 

cap = cv.VideoCapture(0)  # Webcam capture
                          # cap is used later to read frames in a loop

# MediaPipe documentation
# Google      # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# readthedocs # https://mediapipe.readthedocs.io/en/latest/solutions/hands.html 
    
# Initialize MediaPipe hands model  
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize FPS calculation
fps_calc = CvFpsCalc() # Imported over from another project on GitHub -> cvfpscalc.py

# Load the trained model ('asl_model.h5')
model = tf.keras.models.load_model('model/asl_model.h5')

# Label mapping (letters A-Z)
class_labels = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','no_gesture'
]

def extract_keypoints(image):
    """ Extract hand keypoints from user using MediaPipe """
    try:
        # Revise image -> better preformance
        image = cv.resize(image, (640, 480)) # (640, 480) is default webcam resolution
                                             # Consistent input size will help improve speed 
                                             # and stability of landmark detection

        # Convert image to RGB spectrum
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # OpenCV images are by default BGR format
        results = hands.process(image_rgb) # Process image with MediaPipe

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark # Get hand's landmark
                                                                 # returns a list of 21 landmark points for a single hand detected
                                                                 # Each landmark has an x, y, and z coordinate
            keypoints = []
            for landmark in landmarks:
                keypoints.append(landmark.x)  # x-coord
                keypoints.append(landmark.y)  # Y-coord
                keypoints.append(landmark.z)  # Z-coord (depth)
            return keypoints
    except Exception as e:
        print(f"Error processing image: {e}")
    return None

    # Expected output is a list of 63 values (21 landmarks x 3 coordinates) for an image/hand

def classify_gesture(keypoints):
    """ Classify the hand gesture using the trained model. """
    keypoints = np.array([keypoints])
    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

while True:
    ret, frame = cap.read() # One frame
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv.flip(frame, 1) 

    # Convert BGR -> RGB for MediaPipe
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Calculate FPS
    fps = fps_calc.get()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the image
            h, w, _ = frame.shape # _ Represents color channels (RGB), but we don't need it so _ is an unused variable, system crashes without it 
            hand_points = []
            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                hand_points.append((cx, cy))
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw bounding box
            min_x = min(hand_points, key=lambda p: p[0])[0]
            max_x = max(hand_points, key=lambda p: p[0])[0]
            min_y = min(hand_points, key=lambda p: p[1])[1]
            max_y = max(hand_points, key=lambda p: p[1])[1]
            cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            # Draw skeleton
            for connection in mp_hands.HAND_CONNECTIONS:
                start = hand_landmarks.landmark[connection[0]]
                end   = hand_landmarks.landmark[connection[1]]
                start_point = (int(start.x * w), int(start.y * h))
                end_point   = (int(end.x * w), int(end.y * h))
                cv.line(frame, start_point, end_point, (0, 0, 255), 2)

            # Classification
            # Convert the 21 landmarks to the shape (63,)
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            detected_letter = classify_gesture(landmark_list)
            if detected_letter != "no_gesture":  # or remove this check if you want to show everything
                cv.putText(frame, detected_letter, (min_x, min_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show FPS
    cv.putText(frame, f'FPS: {int(fps)}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Hand Gesture Recognition', frame)

    # Exit on 'ESC' key
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
