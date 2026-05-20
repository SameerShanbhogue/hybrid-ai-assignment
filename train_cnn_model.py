import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    # Load image data
    X_train = np.load('data/X_train_images.npy')
    X_val = np.load('data/X_val_images.npy')
    X_test = np.load('data/X_test_images.npy')
    y_train = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')
    y_test = np.load('data/y_test.npy')
    
    # Normalize images
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Build CNN model
    cnn_model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred_probs = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"CNN - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['low', 'medium', 'high']))
    
    # Save model
    cnn_model.save('cnn_model.keras')
    print("Model saved to cnn_model.keras")
    
    return acc, f1, history

if __name__ == "__main__":
    main()