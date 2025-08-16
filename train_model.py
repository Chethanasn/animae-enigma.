import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight

# === Paths ===
data_dir = r"C:\Users\HP\Desktop\gaming\animae_dataset"
model_save_dir = "trained_model"
initial_model_path = os.path.join(model_save_dir, "anime_classifier_mobilenetv2_initial.h5")
fine_tuned_model_path = os.path.join(model_save_dir, "anime_classifier_mobilenetv2_finetuned.h5")
best_model_path = os.path.join(model_save_dir, "anime_classifier_best_model.h5")

# === Image Parameters ===
img_height, img_width = 224, 224
batch_size = 8
initial_epochs = 30
fine_tune_epochs = 20

# === Data Augmentation ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Compute Class Weights ===
class_weights_arr = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}

# === Load MobileNetV2 Base Model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze base model initially

# === Add Custom Classification Head ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# === Compile Model ===
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks (No EarlyStopping) ===
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
    ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

# === Initial Training ===
print("🔁 Training top layers (base model frozen)...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# === Save Initial Model ===
os.makedirs(model_save_dir, exist_ok=True)
model.save(initial_model_path)
print(f"✅ Initial model saved to {initial_model_path}")

# === Fine-Tuning ===
print("\n🔓 Fine-tuning all layers of the base model...")

# Unfreeze the entire base model
base_model.trainable = True

# Recompile with a moderate learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning training
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=fine_tune_epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# === Save Final Fine-Tuned Model ===
model.save(fine_tuned_model_path)
print(f"✅ Fine-tuned model saved to {fine_tuned_model_path}")
print(f"📦 Best model during training saved to {best_model_path}")
