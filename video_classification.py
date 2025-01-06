import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras import mixed_precision  # type: ignore
from sklearn.model_selection import train_test_split

# Optional: Enable mixed precision for faster training on supported GPUs
from tensorflow.keras.mixed_precision import set_global_policy # type: ignore

# Set the mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPU detected. Switching to float32 policy.")
    mixed_precision.set_global_policy('float32')

# Distributed strategy for multiple GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Define paths
    dataset_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset'
    violent_videos_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\Violence'
    nonviolent_videos_path = r'C:\Users\bhilw\OneDrive\Documents\Contentshield\ACCURATE MODEL\Real Life Violence Dataset\NonViolence'

    # Parameters
    frames_per_video = 50  # Increased to 20 frames per video
    img_size = (256, 256)  # Resize to 112x112 pixels
    batch_size = 4  # Reduced batch size for faster training

    # Function to load videos and extract frames from each video
    def load_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(total_frames // frames_per_video, 1)

        for i in range(frames_per_video):  # Ensure this uses frames_per_video = 20
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, img_size)
                frames.append(frame)
            else:
                break
        cap.release()

        if len(frames) < frames_per_video:
            # Padding in case video is shorter than 20 frames
            while len(frames) < frames_per_video:
                frames.append(np.zeros((img_size[0], img_size[1], 3)))

        return np.array(frames)

    # Function to load dataset
    def load_dataset():
        videos = []
        labels = []
        # Load violent videos
        for video_name in os.listdir(violent_videos_path):
            video_path = os.path.join(violent_videos_path, video_name)
            if video_path.endswith('.mp4') or video_path.endswith('.avi'):
                video_frames = load_video(video_path)
                videos.append(video_frames)
                labels.append(1)  # Label for violent videos

        # Load non-violent videos
        for video_name in os.listdir(nonviolent_videos_path):
            video_path = os.path.join(nonviolent_videos_path, video_name)
            if video_path.endswith('.mp4') or video_path.endswith('.avi'):
                video_frames = load_video(video_path)
                videos.append(video_frames)
                labels.append(0)  # Label for non-violent videos

        return np.array(videos), np.array(labels)

    # Loading dataset
    X, y = load_dataset()
    X = X.astype(np.float32) / 255.0  # Ensure X is float32

    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3D DenseNet-inspired block
    def dense_block(x, growth_rate, num_layers):
        for _ in range(num_layers):
            bn = layers.BatchNormalization()(x)
            relu = layers.ReLU()(bn)
            conv = layers.Conv3D(growth_rate, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(relu)
            x = layers.concatenate([x, conv])
        return x

    # Bottleneck transition layer
    def transition_layer(x, reduction):
        bn = layers.BatchNormalization()(x)
        relu = layers.ReLU()(bn)
        conv = layers.Conv3D(int(x.shape[-1] * reduction), kernel_size=(1, 1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-4))(relu)
        pool = layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv)
        return pool

    # 3D CNN Model with dropout and additional regularization
    def build_3d_cnn(input_shape=(20, 112, 112, 3), num_classes=2):
        inputs = layers.Input(shape=input_shape, dtype=tf.float32)  # Set dtype to float32

        # Initial Convolution
        x = layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Dense Blocks with Transition Layers
        x = dense_block(x, growth_rate=32, num_layers=4)  # Reduced growth_rate
        x = transition_layer(x, reduction=0.5)

        x = dense_block(x, growth_rate=32, num_layers=6)
        x = transition_layer(x, reduction=0.5)

        x = dense_block(x, growth_rate=32, num_layers=8)
        x = transition_layer(x, reduction=0.5)
        
        # Final Convolutional Block
        x = layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling3D()(x)
        
        # Add Dropout for regularization
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        return model

    # Build the model
    input_shape = (20, 112, 112, 3)  # Set to 20 frames
    model = build_3d_cnn(input_shape=input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Display model architecture
    model.summary()

    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    # Applying the augmentation on the fly during training
    def augment(video_batch, label_batch):
        video_batch = tf.map_fn(lambda video: data_augmentation(video), video_batch)
        return video_batch, label_batch

    # Augmenting the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(augment).batch(batch_size).shuffle(buffer_size=1000).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath='best_video_classification_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Early stopping to prevent overfitting
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=5,  # Stop after 5 epochs without improvement
        restore_best_weights=True
    )

    # Training the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,  # Increased to 30 epochs
        verbose=1,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Save the final model after training
    model.save('video_model.h5')
