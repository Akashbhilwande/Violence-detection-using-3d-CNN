import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras import regularizers

# Define the path to the dataset
VIDEO_DIR = './dataset/videos/'
IMG_SIZE = (160, 160)  # Resize video frames
FRAME_COUNT = 20       # Decreased number of frames to sample from each video
BATCH_SIZE = 8         # Increased batch size for faster training
AUTOTUNE = tf.data.AUTOTUNE  # For parallel processing with tf.data

# Function to extract frames from videos
def extract_frames(video_path):
    video_path = video_path.decode('utf-8')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return np.zeros((FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

    frames = []
    while len(frames) < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame / 255.0  # Normalize to [0, 1]
        frames.append(frame.astype(np.float32))
    
    cap.release()

    # Pad with zeros if fewer frames were extracted
    while len(frames) < FRAME_COUNT:
        frames.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))

    return np.array(frames, dtype=np.float32)

# Function to augment frames
def augment_frames(frames):
    frames = tf.image.random_flip_left_right(frames)
    frames = tf.image.random_brightness(frames, max_delta=0.1)
    frames = tf.image.random_contrast(frames, lower=0.8, upper=1.2)
    frames = tf.image.random_hue(frames, max_delta=0.05)
    frames = tf.image.random_saturation(frames, lower=0.8, upper=1.2)
    return frames

# Load video data with augmentation
def load_video_data(video_path, label):
    frames = tf.numpy_function(extract_frames, [video_path], tf.float32)
    frames.set_shape((FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3))  # Ensure shape consistency
    augmented_frames = augment_frames(frames)
    return augmented_frames, label

# Create dataset function
def create_dataset(video_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    dataset = dataset.map(load_video_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(video_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.cache()  # Cache the dataset to reduce disk I/O
    return dataset

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

# Set up the strategy for distributed training
strategy = tf.distribute.MirroredStrategy()

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Fine-tuning pre-trained MobileNetV2
with strategy.scope():
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Build the model with L2 regularization and dropout
    model = tf.keras.Sequential([
        Input(shape=(FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.TimeDistributed(base_model),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # L2 regularization
        tf.keras.layers.Dropout(0.5),  # Dropout to prevent overfitting
        tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Use a low learning rate for fine-tuning
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up EarlyStopping and other callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Create datasets with augmentation
train_dataset = create_dataset(X_train_paths, y_train, BATCH_SIZE)
validation_dataset = create_dataset(X_test_paths, y_test, BATCH_SIZE)

# Train the model with augmented data
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save('violence_detection_model.h5')

# Fine-tune on external data (if available)
# You will need to create external_train_dataset and external_validation_dataset
# fine_tune_history = model.fit(
#     external_train_dataset,
#     validation_data=external_validation_dataset,
#     epochs=10,  # Fine-tuning with fewer epochs
#     callbacks=[early_stopping, model_checkpoint]
# )

# Save the fine-tuned model
# model.save('violence_detection_finetuned_model.h5')
