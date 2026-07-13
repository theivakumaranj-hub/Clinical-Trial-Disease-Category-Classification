# Topic: Final Precision-Tuned Pipeline with Columns Normalization

import pandas as pd
import numpy as np
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Initializing NLTK features...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("Ingesting local clinical trial dataset...")
df = pd.read_csv('clinical_trials.csv', low_memory=False)

# Normalize column names to eliminate hidden spaces and case issues
df.columns = df.columns.str.strip().str.lower()

text_col = 'brief_summary'
target_col = 'source_condition_query'

print("Running global deduplication and handling missing entries...")
df = df.dropna(subset=[text_col, target_col])
df = df.drop_duplicates(subset=[text_col])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_medical_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

print("Processing summaries token-by-token...")
df['cleaned_summary'] = df[text_col].apply(clean_medical_text)

label_encoder = LabelEncoder()
df['target_encoded'] = label_encoder.fit_transform(df[target_col])

print("Generating and saving all 5 required analytical charts...")

# Chart 1: Disease Category
plt.figure(figsize=(12, 6))
sns.countplot(y=target_col, data=df, order=df[target_col].value_counts().index[:10], hue=target_col, legend=False, palette='viridis')
plt.title('Top 10 Clinical Trial Disease Categories')
plt.xlabel('Number of Trials')
plt.ylabel('Disease Category')
plt.tight_layout()
plt.savefig('disease_distribution.png')
plt.close()

# Chart 2: Word Frequency
all_words = ' '.join(df['cleaned_summary']).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)
words, counts = zip(*common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), hue=list(words), legend=False, palette='magma')
plt.title('Top 20 Most Frequent Medical Terms')
plt.xlabel('Count')
plt.ylabel('Words')
plt.tight_layout()
plt.savefig('word_frequency.png')
plt.close()

# Chart 3: Phase Distribution
if 'phase' in df.columns:
    plt.figure(figsize=(10, 5))
    phase_data = df['phase'].dropna()
    sns.countplot(x=phase_data, order=phase_data.value_counts().index, hue=phase_data, legend=False, palette='crest')
    plt.title('Distribution of Clinical Trial Phases')
    plt.xlabel('Trial Phase')
    plt.ylabel('Count')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('phase_distribution.png')
    plt.close()
else:
    print("Warning: 'phase' column missing!")

# Chart 4: Study Type Distribution
if 'study_type' in df.columns:
    plt.figure(figsize=(10, 5))
    study_data = df['study_type'].dropna()
    sns.countplot(x=study_data, order=study_data.value_counts().index, hue=study_data, legend=False, palette='flare')
    plt.title('Clinical Trial Distribution by Study Type')
    plt.xlabel('Study Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('study_type_distribution.png')
    plt.close()
    
X = df['cleaned_summary']
y = df['target_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Applying noise-filtered sublinear TF scaling...")
tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training classification model...")
model = LogisticRegression(max_iter=1000, solver='saga', C=5.0)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\n================ MODEL EVALUATION REPORT ================")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print("=========================================================\n")

joblib.dump(model, 'medical_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("All artifacts and new charts successfully exported!")
