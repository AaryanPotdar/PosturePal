import * as tf from '@tensorflow/tfjs';

export async function createPostureModel() {
  // Create a model matching the Python architecture
  const model = tf.sequential();
  
  // First dense layer with 16 units and ReLU activation
  model.add(tf.layers.dense({
    inputShape: [7],  // 7 features as input
    units: 16,
    activation: 'relu'
  }));
  
  // Second dense layer with 16 units and ReLU activation
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu'
  }));
  
  // Output layer with sigmoid activation for 0-1 output
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));
  
  // Compile the model with the same optimizer and loss
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });
  
  return model;
}

// Function to normalize input features
export function normalizeFeatures(features: number[]) {
  // Add your feature normalization logic here
  // This should match the normalization used in your Python training
  return features;
}

// Function to make predictions
export async function predictPosture(model: tf.LayersModel, features: number[]) {
  const normalizedFeatures = normalizeFeatures(features);
  const inputTensor = tf.tensor2d([normalizedFeatures], [1, 7]);
  const prediction = await model.predict(inputTensor) as tf.Tensor;
  const score = prediction.dataSync()[0];
  inputTensor.dispose();
  prediction.dispose();
  return score;
} 