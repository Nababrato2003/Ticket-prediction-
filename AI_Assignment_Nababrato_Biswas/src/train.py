import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
try:
    from preprocessing import load_and_preprocess_data
except ImportError:
    from src.preprocessing import load_and_preprocess_data

def train_and_save_model(data_path, model_dir):
    df = load_and_preprocess_data(data_path)
    
    X = df['Cleaned_Description']
    y = df['Ticket Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training TF-IDF Vectorizer with N-Grams...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    print("Training Logistic Regression Model with Tuned Hyperparameters...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1000)
    model.fit(X_train_vec, y_train)
    
    # Save the models
    os.makedirs(model_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    model_path = os.path.join(model_dir, "ticket_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Models saved to {model_dir}")
    return X_test, y_test, vectorizer, model

if __name__ == "__main__":
    train_and_save_model("../data/tickets.csv", "../model")
