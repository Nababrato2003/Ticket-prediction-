import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    from train import train_and_save_model
except ImportError:
    from src.train import train_and_save_model

def evaluate_model(data_path, model_dir, screenshots_dir):
    # For a complete standalone evaluation, we can retrain or just load test data. 
    # To keep it simple, we'll call train_and_save_model to get the exact test split.
    X_test, y_test, vectorizer, model = train_and_save_model(data_path, model_dir)
    
    print("Evaluating Model...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save confusion matrix plot
    os.makedirs(screenshots_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(screenshots_dir, "confusion_matrix.png"))
    plt.close()
    
    print(f"Confusion matrix saved to {screenshots_dir}")

if __name__ == "__main__":
    evaluate_model("../data/tickets.csv", "../model", "../screenshots")
