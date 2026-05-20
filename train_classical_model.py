import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

def main():
    # Load data
    train_df = pd.read_csv('data/train_df.csv')
    val_df = pd.read_csv('data/val_df.csv')
    test_df = pd.read_csv('data/test_df.csv')
    
    X_train_text = train_df['text'].values
    y_train = train_df['risk'].values
    X_val_text = val_df['text'].values
    y_val = val_df['risk'].values
    X_test_text = test_df['text'].values
    y_test = test_df['risk'].values
    
    # Add tabular features
    X_train_tab = train_df[['tenure', 'prev_complaints']].values
    X_val_tab = val_df[['tenure', 'prev_complaints']].values
    X_test_tab = test_df[['tenure', 'prev_complaints']].values
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_val_tfidf = tfidf.transform(X_val_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()
    
    # Combine with tabular features
    X_train_combined = np.hstack([X_train_tfidf, X_train_tab])
    X_val_combined = np.hstack([X_val_tfidf, X_val_tab])
    X_test_combined = np.hstack([X_test_tfidf, X_val_tab])
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_combined, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Classical ML - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['low', 'medium', 'high']))
    
    # Save model and vectorizer
    joblib.dump({'classifier': clf, 'tfidf': tfidf}, 'classical_model.pkl')
    print("Model saved to classical_model.pkl")
    
    # Return metrics for evaluation report
    return acc, f1

if __name__ == "__main__":
    main()