import pickle
import os
try:
    from preprocessing import clean_text
except ImportError:
    from src.preprocessing import clean_text

def load_models(model_dir):
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(model_dir, "ticket_classifier.pkl")
    
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Models not found. Please run train.py first.")
        
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    return vectorizer, model

def predict_ticket(description, model_dir="../model"):
    vectorizer, model = load_models(model_dir)
    
    cleaned = clean_text(description)
    vec = vectorizer.transform([cleaned])
    
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    conf = max(probs)
    
    return pred, conf

if __name__ == "__main__":
    test_cases = [
        "I am unable to login because my password is not working.",
        "The application crashes every time I try to save."
    ]
    
    for tc in test_cases:
        pred, conf = predict_ticket(tc)
        print(f"Ticket: {tc}")
        print(f"Predicted Category: {pred} (Confidence: {conf*100:.1f}%)\n")
