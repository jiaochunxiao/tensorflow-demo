# CNN Implementation for MNIST Neural Network
# Improved version with convolutional layers

import tensorflow as tf
import argparse
import datetime

# Constants (default values, can be overridden by command line arguments)
IMAGE_SIZE = 28
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN model on MNIST dataset')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train')
    parser.add_argument('--use_data_augmentation', action='store_true', help='Use data augmentation')
    return parser.parse_args()

# Load MNIST dataset
def load_data():
    mnist = tf.keras.datasets.mnist.load_data()
    return mnist

# Create a CNN model
def create_cnn_model():
    # Input layer - reshape to include channel dimension
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    
    # First convolutional block
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Second convolutional block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(10)(x)
    
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
def prepare_data(mnist_data, use_data_augmentation=False):
    (train_images, train_labels), (test_images, test_labels) = mnist_data
    
    # Reshape images to include channel dimension and normalize
    train_images = train_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1).astype('float32') / 255.0
    
    # Create data augmentation if requested
    if use_data_augmentation:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1)
        ])
        
        # Apply data augmentation only to the training images
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
        
        return train_dataset, test_dataset, train_images.shape[0], test_images.shape[0]
    else:
        return (train_images, train_labels), (test_images, test_labels)

# Main function
def main():
    # Parse arguments
    args = parse_args()
    
    # Update constants from arguments
    global BATCH_SIZE, LEARNING_RATE, EPOCHS
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    
    # Load and prepare data
    mnist_data = load_data()
    data = prepare_data(mnist_data, args.use_data_augmentation)
    
    # Create model
    model = create_cnn_model()
    
    # Setup callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001),
        
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            histogram_freq=1
        )
    ]
    
    # Train model
    if args.use_data_augmentation:
        train_dataset, test_dataset, train_size, test_size = data
        steps_per_epoch = train_size // BATCH_SIZE
        validation_steps = test_size // BATCH_SIZE
        
        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
    else:
        (train_images, train_labels), (test_images, test_labels) = data
        
        model.fit(
            train_images, train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(test_images, test_labels),
            callbacks=callbacks
        )
    
    # Evaluate model
    if args.use_data_augmentation:
        test_loss, test_acc = model.evaluate(test_dataset)
    else:
        test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save model
    model.save('mnist_cnn_model')
    print("Model saved to 'mnist_cnn_model'")
    print("Best model saved to 'best_model.h5'")
    print("\nTo visualize training metrics, run: tensorboard --logdir=logs/fit")

if __name__ == "__main__":
    main()
