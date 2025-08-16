import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
data_dir = "../anime_dataset"  # Make sure this path is correct from train_model.py
model_save_path = "trained_model/anime_classifier_model.h5"

# Image parameters
img_height, img_width = 150, 150
batch_size = 16
epochs = 10

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Debug prints before training
print("Class indices:", train_generator.class_indices)
print("Number of training batches:", len(train_generator))
print("Number of validation batches:", len(val_generator))

# Try loading one batch from train_generator to check shapes
x_batch, y_batch = next(train_generator)
print("One batch shape of images:", x_batch.shape)
print("One batch shape of labels:", y_batch.shape)

# Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Save model
os.makedirs("trained_model", exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
