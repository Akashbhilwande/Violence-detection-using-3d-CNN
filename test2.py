import os
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('violence_detection_model.h5')

# Paths to the test videos (you need to specify the correct path)
test_videos_path = 'dataset/videos/test_videos '  # Adjust if necessary

# Parameters
frames_per_video = 10  # Same as used during training
img_size = (112, 112)  # Same as used during training

# Function to load and process test videos
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(total_frames // frames_per_video, 1)

    for i in range(frames_per_video):  # Extracting 10 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
        else:
            break
    cap.release()

    if len(frames) < frames_per_video:
        # Padding in case video is shorter than 10 frames
        while len(frames) < frames_per_video:
            frames.append(np.zeros((img_size[0], img_size[1], 3)))

    return np.array(frames)

# Function to make predictions on a single video
def predict_video(video_path):
    video_frames = load_video(video_path)
    video_frames = np.expand_dims(video_frames, axis=0)  # Add batch dimension
    video_frames = video_frames.astype(np.float32) / 255.0  # Normalize
    predictions = model.predict(video_frames)
    
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

# Loop through test videos and make predictions
for video_name in os.listdir(test_videos_path):
    video_path = os.path.join(test_videos_path, video_name)
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        predicted_class, confidence = predict_video(video_path)
        label = 'Violent' if predicted_class == 1 else 'Non-Violent'
        print(f"Video: {video_name}, Prediction: {label}, Confidence: {confidence:.4f}")
