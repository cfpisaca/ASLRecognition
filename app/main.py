import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from cvfpscalc import CvFpsCalc 
import time

# 1. Initialize webcam capture; 'cap' will be used in a frame-reading loop
cap = cv.VideoCapture(0)  # Webcam capture
                          # cap is used later to read frames in a loop
    
# 2. Initialize the MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. Initialize FPS (frames per second) calculator
fps_calc = CvFpsCalc() 

# 4. Define a custom GCNLayer for loading/using the GCN model
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        # input_shape: (batch_size, num_nodes, input_dim)
        input_dim = input_shape[-1]
        self.w = self.add_weight(shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='w')
        super(GCNLayer, self).build(input_shape)
        
    def call(self, inputs, adj):
        # inputs: (batch_size, num_nodes, input_dim)
        # Multiply inputs by the weight matrix
        x = tf.matmul(inputs, self.w)  # shape: (batch_size, num_nodes, output_dim)
        # Propagate through the graph (adjacency multiplication)
        x = tf.matmul(adj, x)  # shape: (batch_size, num_nodes, output_dim)
        if self.activation is not None:
            x = self.activation(x)
        return x

# 5. Load all trained models (MLP, GCN, CNN).
#    For the GCN model, provide the custom_objects parameter to include GCNLayer.
model_mlp = tf.keras.models.load_model('model/asl_mlp_model.h5')
model_gcn = tf.keras.models.load_model('model/asl_gcn_model.h5', custom_objects={'GCNLayer': GCNLayer})
model_cnn = tf.keras.models.load_model('model/asl_cnn_model.h5')  
model_combined = tf.keras.models.load_model('model/asl_combined_model.h5', custom_objects={'GCNLayer': GCNLayer})

# 6. Choose model type: 'mlp' (multilayer perceptron model), 'gcn' (graph model), or 'cnn' (convolutional model)
MODEL_TYPE = 'combined' 

# 7. Label mapping (letters A-Z, plus space, del, and no_gesture)
class_labels = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','no_gesture','space'
]

# 8. Global variables for recognized text and for stabilizing detections
recognized_text = ""
stable_letter = None
stable_start_time = None
flash_start_time = None  # Time when flash starts
flash_duration = 0.5     # Duration of the flash effect in seconds

# 9. Helper function to classify gestures using the selected model
def classify_gesture(landmarks):
    """ Classify the hand gesture using the selected trained model. """
    if MODEL_TYPE == 'mlp':
        # For the base model, flatten the 21 landmarks (each with x, y, z) into a 63-length vector
        keypoints = []
        for lm in landmarks:
            keypoints.extend(lm)  # lm is a list: [x, y, z]
        keypoints = np.array([keypoints])
        prediction = model_mlp.predict(keypoints)

    elif MODEL_TYPE == 'gcn':
        # For the GCN model, convert landmarks into a (21, 3) array and add batch dimension -> (1, 21, 3)
        keypoints_array = np.array(landmarks)
        keypoints_array = np.expand_dims(keypoints_array, axis=0)
        prediction = model_gcn.predict(keypoints_array)

    elif MODEL_TYPE == 'cnn':
        # For the CNN model on keypoints, reshape (21, 3) -> (21, 3, 1), then batch dimension -> (1, 21, 3, 1)
        keypoints_array = np.array(landmarks)
        keypoints_array = keypoints_array.reshape((1, 21, 3, 1))
        prediction = model_cnn.predict(keypoints_array)

    elif MODEL_TYPE == 'combined':
        # For the combined GCN+CNN model: shape => (1, 21, 3)
        keypoints_array = np.array(landmarks)
        keypoints_array = np.expand_dims(keypoints_array, axis=0)
        prediction = model_combined.predict(keypoints_array)

    else:
        return "no_gesture"

    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

# 10. Generator function to yield frames from the webcam for streaming
# (Modified to show directly in a window instead of yielding)
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
            h, w, _ = frame.shape # _ Represents color channels (RGB)
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
            # For the base model: flatten to (63,)
            # GCN: (21,3)
            # CNN: (21,3,1)
            # Combined: (21,3)
            landmark_list = []
            for lm in hand_landmarks.landmark:
                # Append as a list [x, y, z]
                landmark_list.append([lm.x, lm.y, lm.z])

            detected_letter = classify_gesture(landmark_list)
            if detected_letter != "no_gesture":
                current_time = time.time()
                # If no stable letter, or if different letter detected, reset
                if stable_letter is None or detected_letter != stable_letter:
                    stable_letter = detected_letter
                    stable_start_time = current_time
                else:
                    # Check if the letter has been stable for at least 0.75s
                    if current_time - stable_start_time >= 0.75:
                        if stable_letter == "space":
                            recognized_text += " "
                        elif stable_letter == "del":
                            recognized_text = recognized_text[:-1]
                        else:
                            recognized_text += stable_letter
                        flash_start_time = current_time
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

        # Show recognized sentence
    cv.putText(frame, f'Text: {recognized_text}', (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.imshow("ASL Recognition", frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC key to quit
        break
    elif key == ord('c'):  # 'c' key to clear the recognized text
        recognized_text = ""

cap.release()
cv.destroyAllWindows()
