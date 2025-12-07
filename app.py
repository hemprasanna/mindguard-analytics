# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Sentence-transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SB = True
except Exception:
    HAS_SB = False

# Page config
st.set_page_config(page_title="MindGuard Analytics", layout="wide", initial_sidebar_state="expanded")

# --- Simple CSS for nicer look ---
st.markdown("""
<style>
.main-header { font-size: 2.6rem; font-weight:800; text-align:center; color:#1a1a1a; margin:18px 0; }
.small-muted { color:#666; font-size:0.9rem; text-align:center; }
.card { background:white; padding:18px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Demo dataset loader
# -----------------------
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_demo_data(n=1000):
    np.random.seed(42)
    risk_levels = np.random.choice(['Low','Medium','High'], n, p=[0.55,0.3,0.15])
    texts = []
    for r in risk_levels:
        if r == 'High':
            texts.append(np.random.choice([
                "I want to end it all", "I wish I were dead", "Nobody would miss me",
                "I can't keep going", "I think about killing myself", "Life is pointless"
            ]))
        elif r == 'Medium':
            texts.append(np.random.choice([
                "Feeling hopeless recently", "I am struggling a lot", "Everything seems heavy",
                "I don't know what to do", "I'm very down these days", "I feel alone"
            ]))
        else:
            texts.append(np.random.choice([
                "Had a great day today", "Loving life", "Went for a walk with family",
                "Cooking and relaxed", "Excited about my trip", "Enjoyed a nice coffee"
            ]))
    df = pd.DataFrame({
        'text': texts,
        'risk_level': risk_levels,
        'post_length': [len(t) for t in texts],
        'sentiment_score': np.random.uniform(-1,1,n)
    })
    return df

# -----------------------
# Helpers
# -----------------------
def create_wordcloud(text_series):
    text = " ".join(text_series.astype(str))
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# -----------------------
# Build small, robust classifier using embeddings + logistic regression
# This is cached so training happens only once per session.
# -----------------------
@st.cache_resource(show_spinner=False)
def build_local_classifier():
    """
    Builds a classifier using sentence-transformers embeddings and logistic regression.
    Returns: encoder (LabelEncoder), model (LogisticRegression), embedder (SentenceTransformer)
    """
    if not HAS_SB:
        return None, None, None, "sentence-transformers not installed"
    
    # Curated examples (diverse phrasing) for better generalization
    high_examples = [
        "I want to die", "I wish I was dead", "I don't want to live anymore",
        "The world would be better without me", "I am going to kill myself",
        "I have thought about ending my life", "I can't take it anymore", 
        "I want to end everything", "I will hurt myself", "I feel like I'm done living"
    ]
    med_examples = [
        "I don't feel like living but I have responsibilities", "I feel hopeless",
        "Sometimes I think about not being here", "I am struggling and tired",
        "I feel empty and lost", "I have thoughts about quitting", "I might hurt myself but unsure",
        "I don't know if I can continue", "Lately everything feels meaningless", "I am concerned about my mental health"
    ]
    low_examples = [
        "I want to live and be happy", "I love this life", "Feeling grateful today",
        "I am excited about my future", "I am fine, just busy", "Looking forward to weekend",
        "I enjoyed dinner with family", "I want to travel more", "I love my friends and family", "The weather is great today"
    ]
    # augment with paraphrases / variations (simple)
    def augment(list_of_sentences):
        out = []
        for s in list_of_sentences:
            out.append(s)
            out.append(s + " .")
            out.append(s.replace("I ", "Sometimes I ").replace("I'", "I "))
            # small shuffle of words if possible
            words = s.split()
            if len(words) > 2:
                out.append(" ".join(words[::-1]))
        return list(dict.fromkeys(out))
    high = augment(high_examples)
    med = augment(med_examples)
    low = augment(low_examples)
    
    # Add some negative and positive neutral examples to help unrelated detection
    unrelated = [
        "I want to eat chocolate", "The car broke down", "I like programming",
        "Studying for exams", "Going to the market", "Bought a new phone",
        "Watching a movie tonight", "I cooked pasta today"
    ]
    
    texts = high + med + low + unrelated
    labels = (["High"]*len(high)) + (["Medium"]*len(med)) + (["Low"]*len(low)) + (["Low"]*len(unrelated))
    
    # Shuffle
    texts, labels = shuffle(texts, labels, random_state=42)
    
    # embedder
    embedder = SentenceTransformer("all-MiniLM-L6
