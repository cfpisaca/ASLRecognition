from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from cvfpscalc import CvFpsCalc 
import time

app = Flask(__name__)

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

# Label mapping (letters A-Z, plus space and del, then no_gesture)
class_labels = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','no_gesture','space'
]

# Global variable to hold the recognized text
recognized_text = ""
# Variables for stable detection
stable_letter = None
stable_start_time = None
# Variables for flash effect when a letter is recognized
flash_start_time = None  # Time when flash starts
flash_duration = 0.5     # Duration of the flash effect in seconds

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

def generate_frames():
    global recognized_text, stable_letter, stable_start_time, flash_start_time
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
                    current_time = time.time()
                    # If no stable letter, or if different letter detected, reset
                    if stable_letter is None or detected_letter != stable_letter:
                        stable_letter = detected_letter
                        stable_start_time = current_time
                    else:
                        # Check if the letter has been stable for at least 1.00 seconds
                        if current_time - stable_start_time >= 1.00:
                            # Update recognized_text based on the stable detected gesture
                            if stable_letter == "space":
                                recognized_text += " "
                            elif stable_letter == "del":
                                recognized_text = recognized_text[:-1]
                            else:
                                recognized_text += stable_letter
                            # Set flash start time for visual flash effect within the hand bounding box
                            flash_start_time = current_time
                            # Reset stable detection so it only triggers once per gesture
                            stable_letter = None
                            stable_start_time = None
                    # Overlay the detected gesture near the bounding box
                    cv.putText(frame, detected_letter, (min_x, min_y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Add visual flash effect within the hand bounding box
                    if flash_start_time is not None and time.time() - flash_start_time < flash_duration:
                        overlay = frame.copy()
                        cv.rectangle(overlay, (min_x, min_y), (max_x, max_y), (255, 255, 255), -1)
                        alpha = 0.4  # Transparency factor
                        frame[min_y:max_y, min_x:max_x] = cv.addWeighted(
                            overlay[min_y:max_y, min_x:max_x], alpha,
                            frame[min_y:max_y, min_x:max_x], 1 - alpha, 0
                        )
                    else:
                        flash_start_time = None
                else:
                    stable_letter = None
                    stable_start_time = None

        # Show FPS
        cv.putText(frame, f'FPS: {int(fps)}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # # Overlay the running recognized text at the bottom of the frame
        # h, w, _ = frame.shape
        # cv.putText(frame, recognized_text, (10, h - 10),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame to JPEG
        ret, buffer = cv.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognized_text')
def recognized_text_route():
    return jsonify({'recognized_text': recognized_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global recognized_text
    recognized_text = ""
    return jsonify({'recognized_text': recognized_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
