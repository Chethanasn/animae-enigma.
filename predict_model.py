import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
data_dir = r"C:\Users\HP\Desktop\gaming\animae_dataset"
fine_tuned_model_path = r"trained_model\anime_classifier_mobilenetv2_finetuned.h5"

# === Image parameters ===
img_height, img_width = 224, 224
batch_size = 8

# === Create validation data generator ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Load model ===
model = load_model(fine_tuned_model_path)

# === Predict on validation data ===
val_gen.reset()  # Important to reset generator before prediction
predictions = model.predict(val_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# === True labels ===
true_classes = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# === Confusion matrix ===
cm = confusion_matrix(true_classes, predicted_classes)

# === Plot confusion matrix ===
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# === Classification report ===
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
