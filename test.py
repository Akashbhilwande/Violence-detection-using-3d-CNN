import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
SEQUENCE_LENGTH = 120
BATCH_SIZE = 8

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
            state_dict[key] = torch.tensor(f[key][()])  # Load weights
        model.load_state_dict(state_dict, strict=False)

# Load the test dataset
test_violence_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\Test_new\Test\Violence'
test_nonviolence_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\Test_new\Test\NonViolence'

test_video_paths = []
test_labels = []

for video in os.listdir(test_violence_path):
    if video.endswith('.mp4'):
        test_video_paths.append(os.path.join(test_violence_path, video))
        test_labels.append(1)

for video in os.listdir(test_nonviolence_path):
    if video.endswith('.mp4'):
        test_video_paths.append(os.path.join(test_nonviolence_path, video))
        test_labels.append(0)

test_dataset = VideoDataset(test_video_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the model and load weights
model = VideoModel()
load_weights_from_h5(model, 'video_model.h5')

# Define loss function
criterion = nn.CrossEntropyLoss()

# Initialize metrics
correct = 0
total = 0
test_loss = 0.0
true_positives = 0  # Predicted Violence, Actual Violence
true_negatives = 0  # Predicted Non-Violence, Actual Non-Violence
false_positives = 0  # Predicted Violence, Actual Non-Violence
false_negatives = 0  # Predicted Non-Violence, Actual Violence

# Testing the model
model.eval()
with torch.no_grad():
    for inputs, target, video_path in test_loader:
        inputs = inputs.permute(0, 2, 1, 3, 4)  # Permute input dimensions
        outputs = model(inputs)
        loss = criterion(outputs, target)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Update confusion matrix
        if predicted.item() == 1 and target.item() == 1:
            true_positives += 1
        elif predicted.item() == 0 and target.item() == 0:
            true_negatives += 1
        elif predicted.item() == 1 and target.item() == 0:
            false_positives += 1
        elif predicted.item() == 0 and target.item() == 1:
            false_negatives += 1

        # Print video result
        predicted_label = 'Violence' if predicted.item() == 1 else 'Non-Violence'
        true_label = 'Violence' if target.item() == 1 else 'Non-Violence'
        print(f"Video: {video_path[0]}, Predicted: {predicted_label}, Actual: {true_label}")

# Calculate overall metrics
accuracy = 100 * correct / total
print(f'\nTest Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%')

# Print confusion matrix and performance metrics
print("\nConfusion Matrix:")
print(f"True Positives (Violence): {true_positives}")
print(f"True Negatives (Non-Violence): {true_negatives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\nPerformance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
