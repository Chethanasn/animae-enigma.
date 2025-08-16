import tensorflow as tf

# Change this to the path of your existing model
original_model_path = "your_model.h5"  # <-- use your actual model path
new_model_path = "resaved_model.h5"

# Load using the current (2.16.1) TensorFlow version
model = tf.keras.models.load_model(original_model_path)

# Save it again to make it compatible with 2.16.1
model.save(new_model_path)

print("✅ Model re-saved as:", new_model_path)
