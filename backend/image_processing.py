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

def process_dataset(data_dir, augment=False, is_test=False):
    """
    Process images from a dataset.
    
    For training (is_test=False): data_dir should have subfolders for each class.
    For testing (is_test=True): data_dir contains image files where the label is deduced from the file name.
    """
    images = []
    labels = []

    if is_test:
        # Process test images 
        start_total = time.time()
        image_files = os.listdir(data_dir)
        print(f"Processing {len(image_files)} test images...")
        for img_name in image_files:
            img_path = os.path.join(data_dir, img_name)
            image = cv.imread(img_path)
            if image is None:
                continue

            # Extract keypoints from the original image
            keypoints = extract_keypoints(image)
            if keypoints:
                # Assume the filename is the label with a '_test' suffix (e.g., "A_test.jpg" -> "A")
                label = os.path.splitext(img_name)[0]  # e.g., "A_test"
                if label.endswith("_test"):
                    label = label[:-5]  # remove the "_test" part
                images.append(keypoints)
                labels.append(label)
        end_total = time.time()
        print(f"Total test processing time: {end_total - start_total:.4f} seconds")
    else:
        # Process training images (assumes one subfolder per class)
        total_start = time.time()
        for label in os.listdir(data_dir):
            class_folder = os.path.join(data_dir, label)
            if os.path.isdir(class_folder):
                class_start = time.time()
                class_images = os.listdir(class_folder)
                print(f"Processing {len(class_images)} images for class '{label}'")
                # Process each image in the class folder
                for img_name in class_images:
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
                class_end = time.time()
                print(f"Processing class '{label}' took {class_end - class_start:.4f} seconds")
        total_end = time.time()
        print(f"Total training processing time: {total_end - total_start:.4f} seconds")

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
    
    # Process training data
    train_dir = os.path.join(base_data_dir, "asl_alphabet_train")
    X_train, y_train, le_train = process_dataset(train_dir, augment=True, is_test=False)
    with open("processed_train.pkl", "wb") as f:
        pickle.dump((X_train, y_train, le_train), f)
    print("Training data saved to 'processed_train.pkl'")

    # Process test data
    test_dir = os.path.join(base_data_dir, "asl_alphabet_test")
    X_test, y_test, le_test = process_dataset(test_dir, augment=False, is_test=True)
    with open("processed_test.pkl", "wb") as f:
        pickle.dump((X_test, y_test, le_test), f)
    print("Test data saved to 'processed_test.pkl'")
