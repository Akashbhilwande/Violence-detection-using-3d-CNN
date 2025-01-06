import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import time
import h5py  # Import h5py for saving/loading HDF5
import os

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
SEQUENCE_LENGTH = 120
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Define the video dataset class
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
        frames = np.transpose(frames, (0, 3, 1, 2))  # Change shape to (sequence_length, channels, height, width)
        
        # Convert frames to tensor
        frames = torch.FloatTensor(frames)  # Shape: (sequence_length, 3, height, width)

        return frames, torch.tensor(label, dtype=torch.long)

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

        # Handle cases where there are fewer frames than sequence_length
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

        # Calculate the input size for the fully connected layer dynamically
        self.fc_input_size = self._get_fc_input_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output layer for two classes
        )

    def _get_fc_input_size(self):
        # Create a dummy input to get the size of the flattened output
        dummy_input = torch.zeros(1, 3, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH)
        x = self.conv3d(dummy_input)
        return x.numel()  # Flatten the tensor and return the number of elements

    def forward(self, x):
        x = self.conv3d(x)
        x = x.flatten(start_dim=1)  # Flatten the tensor to pass to the fully connected layer
        x = self.fc(x)
        return x

# Define paths for the dataset
violence_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\Violence'
nonviolence_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\NonViolence'
test_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\Test_new'  # Add your test path here

# Initialize lists to hold video paths and labels
video_paths = []
labels = []

# Collect video paths and labels for training data
print("Collecting video paths and labels for training data...")
for video in os.listdir(violence_path):
    if video.endswith('.mp4'):
        video_paths.append(os.path.join(violence_path, video))
        labels.append(1)  # 1 for Violence

for video in os.listdir(nonviolence_path):
    if video.endswith('.mp4'):
        video_paths.append(os.path.join(nonviolence_path, video))
        labels.append(0)  # 0 for NonViolence

print(f"Total videos collected: {len(video_paths)}")

# Split the dataset into training and validation sets
print("Splitting the dataset into training and validation sets...")
train_video_paths, val_video_paths, train_labels, val_labels = train_test_split(
    video_paths, labels, test_size=0.2, random_state=42
)

# Create datasets and data loaders
print("Creating datasets and data loaders...")
train_dataset = VideoDataset(train_video_paths, train_labels)
val_dataset = VideoDataset(val_video_paths, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test dataset
test_video_paths = []
test_labels = []

# Collect video paths and labels for test data
print("Collecting video paths and labels for test data...")
for video in os.listdir(test_path):
    if video.endswith('.mp4'):
        test_video_paths.append(os.path.join(test_path, video))
        test_labels.append(0)  # Modify this based on your test set

test_dataset = VideoDataset(test_video_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, loss function, and optimizer
model = VideoModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Check the input size for the first Linear layer
print(f"Input size for the first Linear layer: {model.fc_input_size}")

# Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    for batch_idx, (inputs, target) in enumerate(train_loader):
        start_time = time.time()
        inputs = inputs.permute(0, 2, 1, 3, 4)  # Change shape to (batch_size, channels, depth, height, width)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Print batch loss every few batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Validation step
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, target in val_loader:
            inputs = inputs.permute(0, 2, 1, 3, 4)
            outputs = model(inputs)
            val_loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
          f'Train Loss: {epoch_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%')

# Saving the trained model
torch.save(model.state_dict(), 'video_model.pth')
