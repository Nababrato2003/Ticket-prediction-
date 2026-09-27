┌─────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCE                      │
│                  (data/tickets.csv)                      │
│              8,469 Customer Support Tickets              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              STEP 1: DATA PREPROCESSING                 │
│                (src/preprocessing.py)                    │
│                                                         │
│  • Drop missing values (dropna)                         │
│  • Remove duplicate descriptions (drop_duplicates)      │
│  • Lowercase all text                                   │
│  • Remove punctuation (Regex)                           │
│  • Strip extra whitespace                               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            STEP 2: FEATURE ENGINEERING                  │
│              (TF-IDF Vectorizer)                        │
│                                                         │
│  • Convert text → numerical matrix                      │
│  • N-Grams: ngram_range=(1, 2)                          │
│  • Max Features: 5,000                                  │
│  • Remove English stop words                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          STEP 3: TRAIN/TEST SPLIT (80/20)               │
│                                                         │
│  • 80% Training Data → Used to teach the model          │
│  • 20% Testing Data  → Used to evaluate the model       │
│  • random_state = 42 (reproducibility)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            STEP 4: MODEL TRAINING                       │
│               (src/train.py)                            │
│                                                         │
│  • Algorithm: Logistic Regression                       │
│  • Hyperparameters: C=1000, class_weight='balanced'     │
│  • Saves: model/tfidf_vectorizer.pkl                    │
│  • Saves: model/ticket_classifier.pkl                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            STEP 5: MODEL EVALUATION                     │
│              (src/evaluate.py)                          │
│                                                         │
│  • Accuracy:  ~97%                                      │
│  • Precision: ~97%                                      │
│  • Recall:    ~97%                                      │
│  • F1 Score:  ~97%                                      │
│  • Confusion Matrix (16x16)                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 6: STREAMLIT WEB APPLICATION               │
│                   (app.py)                              │
│                                                         │
│  Tab 1: Data Prep & EDA (Charts & Visualizations)       │
│  Tab 2: Model Training & Testing (Metrics Dashboard)    │
│  Tab 3: Predict New Ticket                              │
│                                                         │
│  ┌───────────────────────────────────────────────┐      │
│  │      USER TYPES TICKET DESCRIPTION            │      │
│  │  "My battery drains fast after the update"    │      │
│  └───────────────────┬───────────────────────────┘      │
│                      ▼                                  │
│  ┌───────────────────────────────────────────────┐      │
│  │      ML PREDICTION OUTPUT                     │      │
│  │  Category: Battery life | Confidence: 97%     │      │
│  └───────────────────┬───────────────────────────┘      │
│                      ▼                                  │
│  ┌───────────────────────────────────────────────┐      │
│  │  BONUS: RULE-BASED AUTOMATED RESPONSE         │      │
│  │  "Battery degradation can happen over time.   │      │
│  │   We recommend turning off background apps."  │      │
│  └───────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
