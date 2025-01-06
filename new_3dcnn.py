import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader  
import cv2
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available()) 

# Custom Dataset for Video Frames
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, resize=(128, 128), transform=None):
        """
        Args:
            video_paths (list): List of paths to the video files.
            labels (list): List of labels corresponding to each video.
            num_frames (int): Number of frames to extract from the video.
            resize (tuple): Dimensions to resize the frames.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        """
        Extract frames from the video and resize them.
        Args:
            video_path (str): Path to the video file.
        Returns:
            np.array: Array of frames with shape (num_frames, height, width, channels).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // self.num_frames  # evenly sample frames

        for i in range(0, total_frames, step):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.resize)  # Resize frame
                frames.append(frame)
            if len(frames) == self.num_frames:
                break

        cap.release()

        # Convert frames to numpy array
        frames = np.array(frames)  # Shape: (num_frames, height, width, channels)
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.extract_frames(video_path)
        frames = np.transpose(frames, (3, 0, 1, 2))  # Change shape to (channels, num_frames, height, width)

        frames = np.array([Image.fromarray(frame) for frame in frames])  # Convert each frame to PIL image

        # Apply transformations if any
        if self.transform:
            frames = self.transform(frames)

        # Convert frames to tensor
        frames = torch.tensor(frames, dtype=torch.float32)

        return frames, torch.tensor(label, dtype=torch.long)

# Define transformations (e.g., ToTensor, Normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize if required
])

# Sample video file paths and labels (replace with actual paths)
video_paths = ["path_to_video_1.mp4", "path_to_video_2.mp4"]  # Add actual paths
labels = [0, 1]  # 0 for non-violence, 1 for violence (example)

# Split data into training and testing sets
train_video_paths, test_video_paths, train_labels, test_labels = train_test_split(video_paths, labels, test_size=0.2, random_state=42)

# Create Dataset objects
train_dataset = VideoDataset(video_paths=train_video_paths, labels=train_labels, transform=transform)
test_dataset = VideoDataset(video_paths=test_video_paths, labels=test_labels, transform=transform)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 3D CNN Model (Simple)
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 256)  # Adjust according to input size
        self.fc2 = nn.Linear(256, 2)  # Binary classification (violence vs non-violence)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Model, Loss, Optimizer
model = Simple3DCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.permute(0, 2, 1, 3, 4).cuda()  # Adjust input shape for Conv3D (B, C, T, H, W)
            targets = targets.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()

            # Update weights
            optimizer.step()

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}%")

# Test function
def test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.permute(0, 2, 1, 3, 4).cuda()  # Adjust input shape for Conv3D (B, C, T, H, W)
            targets = targets.cuda()

            # Forward pass
            outputs = model(inputs)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")

# Start training the model
train(model, train_loader, criterion, optimizer, num_epochs=10)

# Test the model after training
#test(model, test_loader)
