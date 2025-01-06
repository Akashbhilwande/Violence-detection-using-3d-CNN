import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
SEQUENCE_LENGTH = 120

# Define the dataset class
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.extract_frames(video_path)
        
        frames = np.array(frames) / 255.0  # Normalize
        frames = np.transpose(frames, (0, 3, 1, 2))  # Shape: (sequence_length, channels, height, width)
        frames = torch.FloatTensor(frames)  # To tensor
        
        return frames, torch.tensor(label, dtype=torch.long), video_path

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frames.append(frame)
        cap.release()

        # Handle cases where there are fewer frames than SEQUENCE_LENGTH
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1] if frames else np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))

        return frames[:SEQUENCE_LENGTH]

# Define the 3D CNN model
class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        
        self.fc_input_size = self._get_fc_input_size()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def _get_fc_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)
            x = self.conv3d(dummy_input)
            return x.numel()

    def forward(self, x):
        x = self.conv3d(x)
        x = x.reshape(x.size(0), -1)  # Flatten using .reshape()
        x = self.fc(x)
        return x
    


# Load the saved weights from .h5 file
def load_weights_from_h5(model, file_path):
    with h5py.File(file_path, 'r') as f:
        state_dict = {}
        for key in f.keys():
            print(f[key]) 
            state_dict[key] = torch.tensor(f[key][()])
        model.load_state_dict(state_dict, strict=False)

# Function to test a single video
def test_single_video(model, video_path):
    # Preprocess the video
    dataset = VideoDataset([video_path], [0])  # Label is a dummy value
    frames, _, _ = dataset[0]
    frames = frames.unsqueeze(0)  # Add batch dimension
    frames = frames.permute(0, 2, 1, 3, 4)  # Permute to match model input dimensions

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = torch.max(outputs.data, 1)

    # Map prediction to label
    predicted_label = 'Violence' if predicted.item() == 1 else 'Non-Violence'
    print(f"Video: {video_path}, Predicted: {predicted_label}")




# Load the model and weights
model = VideoModel()
load_weights_from_h5(model, r'C:\Users\bhilw\OneDrive\Documents\Contentshield\video_classification_model.h5')

# Path to a single video
video_path = r'"C:\Users\bhilw\Downloads\A-Dataset-for-Automatic-Violence-Detection-in-Videos\violence-detection-dataset\violent\cam1\8.mp4"'

# Test the single video
test_single_video(model, video_path)
