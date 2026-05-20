import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    # Load text data
    train_df = pd.read_csv('data/train_df.csv')
    val_df = pd.read_csv('data/val_df.csv')
    test_df = pd.read_csv('data/test_df.csv')
    
    X_train = train_df['text'].values
    y_train = train_df['risk'].values
    X_val = val_df['text'].values
    y_val = val_df['risk'].values
    X_test = test_df['text'].values
    y_test = test_df['risk'].values
    
    # Text vectorization
    max_tokens = 5000
    sequence_length = 100
    
    text_vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=sequence_length,
        standardize='lower_and_strip_punctuation'
    )
    text_vectorizer.adapt(X_train)
    
    # Build RNN (LSTM) model
    rnn_model = models.Sequential([
        text_vectorizer,
        layers.Embedding(input_dim=max_tokens, output_dim=64, mask_zero=True),
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    
    rnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = rnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred_probs = rnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"RNN (LSTM) - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['low', 'medium', 'high']))
    
    # Save model
    rnn_model.save('rnn_model.keras')
    print("Model saved to rnn_model.keras")
    
    return acc, f1, history

if __name__ == "__main__":
    main()