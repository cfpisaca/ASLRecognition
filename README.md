# Honors Thesis - American Sign Language (ASL) Recognition System

This project uses deep learning models to recognize American Sign Language (ASL) gestures in real-time. It captures video from the webcam, processes the frames to detect hand landmarks using **MediaPipe**, and classifies the gestures using a pre-trained **TensorFlow** model. The project supports multiple models, including MLP, CNN, GCN, and a hybrid GCN+CNN. It includes both training and inference pipelines and calculates the frames per second (FPS) of the recognition process for performance evaluation.

## Project Structure

All core files are located in the `app/` directory:

- **`main.py`**: The main script for real-time hand gesture recognition using a webcam. Includes logic for selecting and running one of four trained models.
- **`image_processing.py`**: A script to preprocess and apply MediaPipe/OpenCV to each image in the dataset and extract 3D hand landmarks.
- **`train_model_mlp.py`**: Trains a simple multilayer perceptron (MLP) model using flattened keypoint vectors.
- **`train_model_cnn.py`**: Trains a convolutional neural network (CNN) using reshaped 3D keypoints.
- **`train_model_gcn.py`**: Trains a graph convolutional network (GCN) by modeling hand keypoints as a graph with spatial relationships.
- **`train_model_combined.py`**: Trains a hybrid model combining both CNN and GCN feature representations.
- **`cvfpscalc.py`**: A utility module to calculate and display the real-time frames per second (FPS) of the recognition pipeline.
- **`processed_train.pkl`**: A pickled file containing preprocessed landmark data and labels used for training.
- **`model/`**: Directory that contains saved trained models such as `asl_mlp_model.h5`, `asl_cnn_model.h5`, `asl_gcn_model.h5`, and `asl_combined_model.h5`.
- **`requirements.txt`**: A file listing all required Python packages for the project.

## Dataset

The dataset used for training the models is the **ASL Alphabet** dataset from Kaggle. You can find the dataset by searching for **ASL Alphabet** on [Kaggle Datasets](https://www.kaggle.com/). This dataset consists of images of hands performing different letters of the American Sign Language alphabet, including additional classes like `space`, `del`, and `no_gesture`.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name/app
   ```

2. **Create a virtual environment** (optional but recommended, run inside `app/` folder):

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the ASL dataset** from Kaggle:

   - Go to [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and download the dataset.
   - Extract the dataset and place it in the folder specified in `image_processing.py` under `train_dir`. The path should be something like:

     ```
     /Users/carlopisacane/Desktop/HonorsThesis/data/asl_alphabet_train
     ```

## Running the Gesture Recognition System

1. **Navigate to the `app/` folder** and activate the virtual environment (if not already activated):

   ```bash
   cd app
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

2. **Run the `main.py` script** to start the real-time gesture recognition system:

   ```bash
   python main.py
   ```

3. The script will:
   - Capture video from your webcam
   - Detect and track hand landmarks using **MediaPipe**
   - Classify the hand gesture using the selected model (`mlp`, `cnn`, `gcn`, or `combined`)
   - Display the recognized letter, the full recognized text, and the FPS in an OpenCV window

4. You can modify the `MODEL_TYPE` variable at the top of `main.py` to choose which model to run.

5. Controls:
   - Press **`ESC`** to exit the app
   - Press **`C`** to clear the recognized sentence

## Training Models (Pre-trained models already saved)

1. **Run the image processing script** to extract keypoints (from inside the `app/` folder):

   ```bash
   python image_processing.py
   ```

2. **Run one or more training scripts** to train different models:

   ```bash
   python train_model_mlp.py         # Train a multilayer perceptron
   python train_model_cnn.py         # Train a CNN model
   python train_model_gcn.py         # Train a GCN model
   python train_model_combined.py    # Train a combined GCN + CNN model
   ```

3. Each training script will:
   - Load the extracted keypoint data from `processed_train.pkl`
   - Split the data into training and validation sets
   - Train the respective model for 10 epochs
   - Save the model to the `model/` directory

## Requirements

- **Python 3.10**
- **Packages** listed in `requirements.txt`:
  - **mediapipe**
  - **opencv-python**
  - **tensorflow**
  - **numpy**
  - **scikit-learn**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.