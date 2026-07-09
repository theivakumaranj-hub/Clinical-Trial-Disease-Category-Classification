# Topic: Clean Production Multi-Chart Dashboard (Vertical Layout)
import streamlit as st
import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

@st.cache_resource
def load_production_artifacts():
    model = joblib.load('medical_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, tfidf, le

model, tfidf, le = load_production_artifacts()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_input_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

st.set_page_config(page_title="Medical Text Classifier", layout="wide")
st.title("🏥 Clinical Trial Disease Category Classification App")
st.markdown("An intelligent system to automatically predict disease categories based on medical text content.")

tab1, tab2 = st.tabs(["🔮 Real-Time Prediction Dashboard", "📊 Dataset EDA Insights"])

with tab1:
    st.subheader("Predict Disease Category from Clinical Summary")
    user_text = st.text_area("Paste the 'Brief Summary' of a medical trial here:", height=200, key="main_text_input")
    
    if st.button("Run Model Prediction", type="primary", key="predict_btn"):
        if user_text.strip() == "":
            st.warning("Please input a valid medical text summary.")
        else:
            cleaned = clean_input_text(user_text)
            vectorized = tfidf.transform([cleaned])
            encoded_pred = model.predict(vectorized)[0]
            predicted_class = le.inverse_transform([encoded_pred])[0]
            
            st.success("Analysis Complete!")
            st.metric(label="Predicted Disease Category Mapping:", value=str(predicted_class).upper())

with tab2:
    st.subheader("Comprehensive Exploratory Data Analysis (EDA) Dashboard")
    st.markdown("Statistical distribution metrics compiled from the raw 160MB clinical trial dataset records.")
    
    # Chart 1: Disease Categories
    st.markdown("---")
    st.subheader("1. 📌 Top Verified Disease Categories")
    if os.path.exists('disease_distribution.png'):
        st.image('disease_distribution.png', use_container_width=True, caption='Distribution of the most dominant disease categories within the dataset.')
    else:
        st.info("Disease distribution chart file not found.")
        
    # Chart 2: Word Frequency
    st.markdown("---")
    st.subheader("2. 📌 Most Frequent Medical Terms")
    if os.path.exists('word_frequency.png'):
        st.image('word_frequency.png', use_container_width=True, caption='Top 20 most frequent text tokens parsed across all trial summaries.')
    else:
        st.info("Word frequency chart file not found.")
        
    # Chart 3: Trial Phases
    st.markdown("---")
    st.subheader("3. 📌 Clinical Trial Phase Distribution")
    if os.path.exists('phase_distribution.png'):
        st.image('phase_distribution.png', use_container_width=True, caption='Breakdown of clinical trials across structural phases (Phase 1 to Phase 4).')
    else:
        st.info("Phase distribution chart missing. Run data_pipeline.py first.")
        
    # Chart 4: Study Types
    st.markdown("---")
    st.subheader("4. 📌 Clinical Trial Distribution by Study Design Type")
    if os.path.exists('study_type_distribution.png'):
        st.image('study_type_distribution.png', use_container_width=True, caption='Comparative volume tracking interventional versus observational study frameworks.')
    else:
        st.info("Study type distribution chart missing. Run data_pipeline.py first.")