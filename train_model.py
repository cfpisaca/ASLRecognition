import os
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout#, Flaten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# MediaPipe hands & extract_keypoints from main.py
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

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

def prepare_data(data_dir, num_samples=None):
    """ Prepare the training data """
    images = []
    labels = []
    sample_count = 0

    # Iterate through each class/letter folder
    for label in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, label)
        if os.path.isdir(class_folder):
            class_images = os.listdir(class_folder)
            print(f"Processing {len(class_images)} images for class {label}")

            # Process each image in the class folder
            for img_name in class_images:
                img_path = os.path.join(class_folder, img_name)
                image = cv2.imread(img_path) # Reads the image with OpenCV
                
                if image is not None:
                    keypoints = extract_keypoints(image)  # MediaPipe landmarks applied to image if hand detected 
                    if keypoints:
                        images.append(keypoints) 
                        labels.append(label)
                        sample_count += 1
                        
                        # Optionally stop after 'num_samples' images for quick testing
                        if num_samples and sample_count >= num_samples:
                            break

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Total images loaded: {len(images)}")

    # Encode labels as integers
    label_encoder = LabelEncoder() # sklearn
    labels = label_encoder.fit_transform(labels)
    
    return images, labels, label_encoder

# Prepare the data
data_dir = '/Users/carlopisacane/Desktop/Honors/HonorsThesis/data/asl_alphabet_train'  # Path to your dataset
X, y, label_encoder = prepare_data(data_dir)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Training set (X_train, y_train)
# Validation set (X_val, y_val)   

# test_size=0.2 means 20% of data goes to validation and 80% remains for training model
# random_state=42 ensures that the split is reproducible

# You need validation because the model is always adjusting weights to minimize loss,
# you risk overfitting the model if only track performance on data being trained.

# In model.fit -> the validation data is set at the end of each epoch and reports the validation
# loss and accuracy. 

# Build the model with regularization
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'), # Input layer and first dense layer
    Dropout(0.5), # Sets 50% of inputs to zero, help prevent overfitting
    Dense(64, activation='relu'), # Second dense layer
    Dropout(0.5), # Another dropout layer
    Dense(26, activation='softmax') # Softmax outputs probability distribution across all 26 classes
])

# Compile the model
model.compile(
    optimizer='adam', # Adapts the learning rate for each parameter using Adam
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_train, y_train, 
    epochs=10, 
    validation_data=(X_val, y_val)
)

model.save('asl_model.h5')  # Save the trained model