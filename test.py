import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import absl.logging

sys.stderr = open(os.devnull, 'w')

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow and absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = all logs, '1' = warnings, '2' = errors, '3' = fatal
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

# Set parameters
IMG_SIZE = (64, 64)  # Resize video frames
FRAME_COUNT = 20      # Number of frames to sample from each video

# Function to extract frames from a video
def extract_frames(V_4):
    cap = cv2.VideoCapture(V_4)
    frames = []
    while len(frames) < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Load the trained model
model = tf.keras.models.load_model('violence_detection_model_with_early_stopping.h5')

# Function to predict if a video is violent or non-violent
def predict_video(video_path):
    frames = extract_frames(video_path)
    
    if frames.shape[0] < FRAME_COUNT:
        print(f"Not enough frames extracted from the video. Extracted {frames.shape[0]} frames.")
        return

    # Normalize and reshape the data
    frames = frames / 255.0
    frames = frames.reshape(-1, FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3)

    # Make prediction
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probabilities = predictions[0]  # Get probabilities for the first sample

    # Map the predicted class to labels
    labels = ['Violent', 'Non-Violent']
    print(f'The video is predicted to be: {labels[predicted_class]}')
    print(f'Predicted probabilities: {predicted_probabilities}')


video_name = r'C:\Users\bhilw\Downloads\Human Fights.mp4'  # The name of your video file
video_path = os.path.join(os.getcwd(), video_name)  # Construct the full path
predict_video(video_path)

