# HonorsThesis

This project uses a deep learning model to recognize American Sign Language (ASL) gestures in real-time. It captures video from the webcam, processes the frames to detect hand landmarks using **MediaPipe**, and classifies the gestures using a pre-trained **TensorFlow** model. The project includes training and inference pipelines and calculates the frames per second (FPS) of the recognition process for performance evaluation.

## Project Structure

- **`main.py`**: The main script for real-time hand gesture recognition using a webcam.
- **`train_model.py`**: A script to train a TensorFlow model to classify ASL gestures using hand landmarks.
- **`cvfpscalc.py`**: A utility script to calculate and display the frames per second (FPS) of the recognition pipeline.
- **`requirements.txt`**: A file listing the required Python packages for the project.

## Dataset

The dataset used for training the model is the **ASL Alphabet** dataset from Kaggle. You can find the dataset by searching for **ASL Alphabet** on [Kaggle Datasets](https://www.kaggle.com/search?q=asl+in%3Adatasets). This dataset consists of images of hands performing different letters of the American Sign Language alphabet.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the ASL dataset** from Kaggle:

   - Go to [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/search?q=asl+in%3Adatasets) and download the dataset.
   - Extract the dataset and place it in the folder specified in `train_model.py` under `data_dir`. The path should be something like:

     ```
     /Users/carlopisacane/Desktop/HonorsThesis/data/asl_alphabet_train
     ```

## Training the Model

1. **Run the training script** to train the model:

   ```bash
   python train_model.py
   ```

2. The script will:
   - Process the images in the dataset and extract the hand landmarks.
   - Split the data into training and validation sets.
   - Build a deep learning model with two **Dense** layers and **Dropout** for regularization.
   - Train the model on the extracted keypoints for 10 epochs.
   - Save the trained model to a file named `asl_model.h5`.

## Running the Gesture Recognition System

1. **Run the `main.py` script** to start the real-time gesture recognition system:

   ```bash
   python main.py
   ```

2. The script will:
   - Capture video from your webcam.
   - Detect and track hand landmarks using **MediaPipe**.
   - Classify the hand gesture using the trained model.
   - Display the recognized gesture and FPS on the screen in real-time.

3. To exit the application, press the **`ESC`** key.

## Requirements

- **Python 3.x**
- **Packages** listed in `requirements.txt`:
  - **mediapipe==0.8.4**
  - **opencv-python==4.6.0.66**
  - **tensorflow==2.9.0**
  - **numpy==1.21.2**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.