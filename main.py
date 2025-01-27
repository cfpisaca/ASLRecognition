import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from cvfpscalc import CvFpsCalc  

cap = cv.VideoCapture(0) # Webcam capture 
                         # cap is used later to read frames in a loop

# MediaPipe documentation
# Google      # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# readthedocs # https://mediapipe.readthedocs.io/en/latest/solutions/hands.html 
    
# Initialize MediaPipe hands model    
mp_hands = mp.solutions.hands                       
hands = mp_hands.Hands( 
    static_image_mode=False,  
    max_num_hands=1,
    min_detection_confidence=0.7, # Higher confidence is worse in poor lightning
    min_tracking_confidence=0.6
)

# Initialize FPS calculation
fps_calc = CvFpsCalc() # Imported over from another project on GitHub -> cvfpscalc.py

# Load the trained model ('asl_model.h5')
model = tf.keras.models.load_model('asl_model.h5')

# Label mapping (letters A-Z)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'no_gesture']

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
    """ Classify the hand gesture using the trained model """
    keypoints = np.array([keypoints])  # Convert keypoints to numpy array to fit the model input
    keypoints = keypoints.reshape(1, -1)  # Flatten the keypoints to a 1D array for the model input
    prediction = model.predict(keypoints) # Model prediction
    predicted_class = np.argmax(prediction) # Find the class with with highest prob.
    return class_labels[predicted_class] # Return label mapping of predicted class

while True:
    ret, image = cap.read() # One frame
    if not ret:
        print("Error: Failed to capture image.")
        break

    image = cv.flip(image, 1)  # Flip the image for mirror view -> for some reason works better for a-b-c-d-e-f (on left hand)

    # Convert the image to RGB for MediaPipe processing
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb) 
    image_rgb.flags.writeable = True

    # Calculate FPS
    fps = fps_calc.get()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the image
            h, w, _ = image.shape # _ Represents color channels (RGB), but we don't need it so _ is an unused variable, system crashes without it 
            hand_landmarks_list = []
            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                hand_landmarks_list.append((cx, cy))  # Store landmarks coordinates
                cv.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Draw the landmark dots

            # Create a bounding box around the hand
            min_x = min(hand_landmarks_list, key=lambda item: item[0])[0]
            max_x = max(hand_landmarks_list, key=lambda item: item[0])[0]
            min_y = min(hand_landmarks_list, key=lambda item: item[1])[1]
            max_y = max(hand_landmarks_list, key=lambda item: item[1])[1]

            # Draw the bounding box 
            cv.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  

            # Get the landmarks for classification
            landmark_list = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
            detected_letter = classify_gesture(landmark_list)

            # Draw the hand skeleton
            for connection in mp_hands.HAND_CONNECTIONS:
                start = hand_landmarks.landmark[connection[0]]
                end = hand_landmarks.landmark[connection[1]]
                start_point = int(start.x * w), int(start.y * h)
                end_point = int(end.x * w), int(end.y * h)
                cv.line(image, start_point, end_point, (0, 0, 255), 2)

            # Display the detected letter above the bounding box
            if detected_letter != "no_gesture":
                cv.putText(image, f'{detected_letter}', (min_x, min_y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display FPS
    cv.putText(image, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed image
    cv.imshow('Hand Gesture Recognition', image)

    # Exit when 'ESC' is pressed
    if cv.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close the window
cap.release()
cv.destroyAllWindows()
