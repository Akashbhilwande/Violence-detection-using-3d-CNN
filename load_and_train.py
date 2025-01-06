import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the pre-built model
model = load_model('video_classification_model.h5')

# Display model architecture (optional)
model.summary()

# Load your dataset (you already have this code)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# Applying the augmentation on the fly during training
def augment(video_batch, label_batch):
    video_batch = tf.map_fn(lambda video: data_augmentation(video), video_batch)
    return video_batch, label_batch

# Augmenting the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(augment).batch(16).shuffle(buffer_size=1000)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)

# Checkpoint callback to save the best model during training
checkpoint_callback = ModelCheckpoint(
    filepath='best_video_classification_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, verbose=1, callbacks=[checkpoint_callback])

# Save the final model after training
model.save('video_classification_trained_model.h5')
