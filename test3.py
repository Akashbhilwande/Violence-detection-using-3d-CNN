import os
import cv2
import numpy as np
import tensorflow as tf

# Set parameters
IMG_SIZE = (64, 64)  # Resize video frames
FRAME_COUNT = 30     # Number of frames to sample from each video

# Function to extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
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
        return None

    # Normalize and reshape the data
    frames = frames / 255.0
    frames = frames.reshape(-1, FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3)

    # Make prediction
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probabilities = predictions[0]  # Get probabilities for the first sample

    # Map the predicted class to labels (reversed logic)
    labels = ['Violent', 'Non-Violent']  # 0 -> Violent, 1 -> Non-Violent
    return predicted_class, predicted_probabilities

# Function to evaluate all videos in a folder
def evaluate_model(test_videos_folder):
    correct_predictions = 0
    total_videos = 0

    for video_file in os.listdir(test_videos_folder):
        video_path = os.path.join(test_videos_folder, video_file)

        # Only process video files
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        # Predict class for the video
        result = predict_video(video_path)

        if result is None:
            continue

        predicted_class, predicted_probabilities = result
        
        # Determine the actual class based on the filename (reversed logic)
        actual_class = 0 if video_file.startswith('V_') else 1  # V_ -> Non-Violent, others -> Violent

        total_videos += 1
        if predicted_class == actual_class:
            correct_predictions += 1

        # Print the result for each video
        predicted_label = 'Non-Violent' if predicted_class == 1 else 'Violent'
        actual_label = 'Non-Violent' if actual_class == 1 else 'Violent'
        print(f"{video_file}: Prediction: {predicted_label}, Actual: {actual_label}")
        print(f"Predicted probabilities: {predicted_probabilities}")

    # Calculate accuracy
    accuracy = (correct_predictions / total_videos) * 100 if total_videos > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Define the path to your test videos folder
test_videos_folder = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\dataset\videos\test_videos'

# Evaluate the model on the test videos
evaluate_model(test_videos_folder)
