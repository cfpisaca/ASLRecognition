import os
import time
import numpy as np
import cv2 as cv
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import pickle

# MediaPipe hands & extract_keypoints from main.py
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(image):
    """ Extract hand keypoints from an image using MediaPipe """
    try:
        # Resize image -> better performance
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

def process_dataset(data_dir, augment=False):
    """
    Process images from a dataset.
    
    For training: data_dir should have subfolders for each class.
    """
    images = []
    labels = []

    for label in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, label)
        if os.path.isdir(class_folder):
            print(f"Processing images for class '{label}'")
            image_files = os.listdir(class_folder)
            for img_name in image_files:
                img_path = os.path.join(class_folder, img_name)
                image = cv.imread(img_path)
                if image is None:
                    continue

                # Extract keypoints from the original image
                keypoints = extract_keypoints(image)
                if keypoints:
                    images.append(keypoints)
                    labels.append(label)

                    # If augment is True, also flip the image horizontally and extract again
                    if augment:
                        flipped_img = cv.flip(image, 1)  # Flip horizontally
                        flipped_keypoints = extract_keypoints(flipped_img)
                        if flipped_keypoints:
                            images.append(flipped_keypoints)
                            labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Total images processed (including augmentation if used): {len(images)}")

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels, label_encoder

if __name__ == "__main__":
    base_data_dir = "/Users/carlopisacane/Desktop/Honors/HonorsThesis/data"
    train_dir = os.path.join(base_data_dir, "asl_alphabet_train")
    
    # Process training data with augmentation enabled
    X_train, y_train, le_train = process_dataset(train_dir, augment=True)
    with open("processed_train.pkl", "wb") as f:
        pickle.dump((X_train, y_train, le_train), f)
    print("Training data saved to 'processed_train.pkl'")
