import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

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
def build_3d_cnn(input_shape=(16, 112, 112, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Dense Blocks with Transition Layers
    x = dense_block(x, growth_rate=32, num_layers=4)
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

# Build and save the model
input_shape = (16, 112, 112, 3)
model = build_3d_cnn(input_shape=input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model architecture and weights
model.save('video_classification_model.keras')

