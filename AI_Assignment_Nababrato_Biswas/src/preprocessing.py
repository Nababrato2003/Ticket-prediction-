import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Ticket Description', 'Ticket Category'])
    df = df.drop_duplicates(subset=['Ticket Description']).copy()
    df['Cleaned_Description'] = df['Ticket Description'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data("../data/tickets.csv")
    print("Data preprocessed successfully.")
    print("Shape:", df.shape)
