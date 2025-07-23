# Simple MNIST Neural Network Implementation
# Based on TensorFlow examples

import tensorflow as tf

# Constants
IMAGE_SIZE = 28
HIDDEN_SIZE = 500
NUM_LABELS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01
MAX_STEPS = 1000

# Load MNIST dataset
def load_data():
    mnist = tf.keras.datasets.mnist.load_data()
    return mnist

# Create a simple neural network model
def create_model():
    # Input layer
    inputs = tf.keras.Input(shape=(IMAGE_SIZE * IMAGE_SIZE,))
    
    # Hidden layer
    hidden = tf.keras.layers.Dense(
        HIDDEN_SIZE, 
        activation='relu',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)
    )(inputs)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        NUM_LABELS,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)
    )(hidden)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with numerically stable loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

# Prepare data for training
def prepare_data(mnist_data):
    (train_images, train_labels), (test_images, test_labels) = mnist_data
    
    # Normalize and reshape images
    train_images = train_images.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).astype('float32') / 255.0
    test_images = test_images.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).astype('float32') / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

# Main function
def main():
    # Load and prepare data
    mnist_data = load_data()
    (train_images, train_labels), (test_images, test_labels) = prepare_data(mnist_data)
    
    # Create model
    model = create_model()
    
    # Train model
    model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=5,
        validation_data=(test_images, test_labels)
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save model
    model.save('mnist_model')
    print("Model saved to 'mnist_model'")

if __name__ == "__main__":
    main()
