import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os

model_path = 'src/model/posture_model.h5'

# Verify model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

print(f"Loading model from {model_path}")
print(f"File size: {os.path.getsize(model_path)} bytes")

# Load the model with error handling for custom objects
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except ValueError as e:
    print("Error loading model. Check if custom_objects are needed.")
    raise e

# Print model summary
print("\nOriginal Model Summary:")
model.summary()

# Create a new model with explicit input layer
inputs = tf.keras.Input(shape=(7,), name='input_1')
x = tf.keras.layers.Dense(16, activation='relu', name='dense_1')(inputs)
x = tf.keras.layers.Dense(16, activation='relu', name='dense_2')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')(x)

new_model = tf.keras.Model(inputs=inputs, outputs=outputs)

print("\nNew Model Summary:")
new_model.summary()

# Get the dense layers from both models
original_dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
new_dense_layers = [layer for layer in new_model.layers if isinstance(layer, tf.keras.layers.Dense)]

# Copy weights from original dense layers to new dense layers
for orig_layer, new_layer in zip(original_dense_layers, new_dense_layers):
    new_layer.set_weights(orig_layer.get_weights())
    print(f"Copied weights from {orig_layer.name} to {new_layer.name}")

# Compile the model
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

# Convert and save the model
output_dir = 'public/assets'
os.makedirs(output_dir, exist_ok=True)

try:
    tfjs.converters.save_keras_model(new_model, output_dir)
    print(f"\nModel converted and saved to {output_dir}/")
except Exception as e:
    print(f"Error during conversion: {e}")
    raise e

# Verify the converted files exist
expected_files = {'model.json', 'group1-shard1of1.bin'}
actual_files = set(os.listdir(output_dir))

if expected_files.issubset(actual_files):
    print("Conversion successful! Files created:")
    for file in actual_files:
        print(f"- {file}")
else:
    print("Warning: Some files are missing after conversion!")
