import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

VIDEO_DIR = './dataset/videos/'
# Prepare the dataset
video_paths = []
y = []

# Process non-violent videos
folder = 'non_violent'
folder_path = os.path.join(VIDEO_DIR, folder)
for video in os.listdir(folder_path):
    video_path = os.path.join(folder_path, video)
    video_paths.append(video_path)
    y.append(0)  # Label for non-violent

# Process violent videos
folder = 'violent'
folder_path = os.path.join(VIDEO_DIR, folder)
for video in os.listdir(folder_path):
    video_path = os.path.join(folder_path, video)
    video_paths.append(video_path)
    y.append(1)  # Label for violent

# Convert to numpy arrays
y = np.array(y)

# Check if any video paths were found
if not video_paths:
    raise ValueError("No video samples found. Please check the dataset.")

# Split into training and test sets
X_train_paths, X_test_paths, y_train, y_test = train_test_split(video_paths, y, test_size=0.2, random_state=42)


# Define the target directory for the test files
TEST_VIDEO_DIR = './test_videos/'
X_train_paths, X_test_paths, y_train, y_test = train_test_split(video_paths, y, test_size=0.2, random_state=42)
# Ensure the target directory exists
if not os.path.exists(TEST_VIDEO_DIR):
    os.makedirs(TEST_VIDEO_DIR)

# Copy the test video files to the new folder
for video_path in X_test_paths:
    # Get the video file name
    video_name = os.path.basename(video_path)
    
    # Define the destination path
    destination_path = os.path.join(TEST_VIDEO_DIR, video_name)
    
    # Copy the file
    shutil.copy(video_path, destination_path)

print(f"Copied {len(X_test_paths)} test videos to {TEST_VIDEO_DIR}")
