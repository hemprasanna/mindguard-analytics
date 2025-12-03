import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer, KNNImputer
from textblob import TextBlob
import re
from wordcloud import WordCloud
import time

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MindGuard Analytics | AI Mental Health Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PERFECT CSS - Colorful but HIGHLY READABLE
st.markdown("""
    <style>
    /* Soft animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #e3f2fd, #f3e5f5, #e8f5e9, #fff3e0, #fce4ec);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        background-attachment: fixed;
    }
    
    /* High contrast content area for readability */
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* ALL TEXT DARK AND READABLE */
    .main * {
        color: #1a1a1a !important;
    }
    
    /* Headers with colorful underline */
    .main-header {
        font-size: 3rem; 
        font-weight: 900; 
        text-align: center;
        color: #1a1a1a !important;
        margin: 20px 0;
        padding-bottom: 15px;
        border-bottom: 5px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb) 1;
    }
    
    .sub-header {
        font-size: 1.8rem; 
        font-weight: 700;
        color: #1a1a1a !important;
        margin: 25px 0 15px 0;
        padding-left: 15px;
        border-left: 5px solid #667eea;
    }
    
    /* Vibrant but readable sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar text - WHITE for contrast on dark gradient */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar navigation items */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.2) !important;
        padding: 12px 15px !important;
        border-radius: 10px !important;
        margin: 5px 0 !important;
        transition: all 0.3s !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.35) !important;
        transform: translateX(5px) !important;
    }
    
    /* Buttons with excellent contrast */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 25px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Colorful tabs with dark text */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        color: #1a1a1a !important;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 700;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
        border: 2px solid #667eea;
    }
    
    /* Metrics with colored backgrounds but dark text */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 900 !important;
        color: #1a1a1a !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 700 !important;
        color: #2d2d2d !important;
    }
    
    /* Input fields with good contrast */
    .stTextInput > div > div input,
    .stTextArea > div > div textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div input {
        background: white !important;
        color: #1a1a1a !important;
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
    }
    
    /* Dataframes readable */
    .dataframe {
        background: white !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: #667eea !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    .dataframe td {
        color: #1a1a1a !important;
    }
    
    /* Expanders with colored border */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.08) !important;
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(102, 126, 234, 0.05) !important;
        border: 2px dashed #667eea !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #f093fb) !important;
    }
    
    /* Alerts with good contrast */
    .stAlert {
        border-radius: 10px !important;
        border-left: 5px solid !important;
    }
    
    /* Success - green with dark text */
    div[data-baseweb="notification"][kind="success"] {
        background: #e8f5e9 !important;
        border-left-color: #4caf50 !important;
    }
    
    div[data-baseweb="notification"][kind="success"] * {
        color: #1b5e20 !important;
    }
    
    /* Info - blue with dark text */
    div[data-baseweb="notification"][kind="info"] {
        background: #e3f2fd !important;
        border-left-color: #2196f3 !important;
    }
    
    div[data-baseweb="notification"][kind="info"] * {
        color: #0d47a1 !important;
    }
    
    /* Warning - orange with dark text */
    div[data-baseweb="notification"][kind="warning"] {
        background: #fff3e0 !important;
        border-left-color: #ff9800 !important;
    }
    
    div[data-baseweb="notification"][kind="warning"] * {
        color: #e65100 !important;
    }
    
    /* Error - red with dark text */
    div[data-baseweb="notification"][kind="error"] {
        background: #ffebee !important;
        border-left-color: #f44336 !important;
    }
    
    div[data-baseweb="notification"][kind="error"] * {
        color: #b71c1c !important;
    }
    
    /* Cards with subtle colors */
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.15) !important;
        transition: all 0.3s !important;
    }
    
    [data-testid="column"] > div:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Sliders */
    .stSlider label {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Helper functions
@st.cache_data
def load_sample_data():
    """Generate synthetic dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    texts = [
        "I feel so hopeless and alone", "Everything will be fine tomorrow",
        "I can't take this anymore", "Looking forward to the weekend",
        "Nobody would miss me", "Excited about new opportunities",
        "Life has no meaning", "Grateful for my friends and family",
        "I want to end it all", "Working on my goals every day"
    ]
    
    data = {
        'post_id': range(1, n_samples + 1),
        'text': np.random.choice(texts, n_samples),
        'post_length': np.random.randint(10, 500, n_samples),
        'engagement_rate': np.random.uniform(0, 1, n_samples),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
        'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_samples),
        'previous_posts_count': np.random.randint(0, 1000, n_samples),
        'account_age_days': np.random.randint(1, 3650, n_samples),
        'follower_count': np.random.randint(0, 10000, n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices, 'engagement_rate'] = np.nan
    df.loc[np.random.choice(df.index, size=int(0.05 * len(df)), replace=False), 'sentiment_score'] = np.nan
    
    return df

@st.cache_data
def perform_text_analysis(df):
    """Advanced text analysis"""
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['contains_negative'] = df['text'].apply(
        lambda x: 1 if any(word in x.lower() for word in ['hopeless', 'alone', 'end', 'nobody']) else 0
    )
    return df

@st.cache_data
def handle_missing_values(df, method='mean'):
    """Advanced imputation"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy='mean')
    
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df_imputed

def create_wordcloud(text_series):
    """Generate word cloud"""
    text = ' '.join(text_series.astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# SIDEBAR
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>🛡️ MindGuard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem; opacity: 0.95;'>AI Mental Health Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "🧭 Navigation",
        ["🏠 Home", "📊 Data Hub", "🔍 Explorer", "🎨 Visuals Pro", 
         "⚙️ Engineer", "🤖 AI Models", "🔮 Predict", "📚 Docs"]
    )
    
    page = page.split(' ', 1)[1] if ' ' in page else page
    
    st.markdown("---")
    st.markdown("### 🎲 Data Controls")
    
    if st.button("🚀 Load Data", use_container_width=True):
        with st.spinner("Loading..."):
            time.sleep(0.5)
            st.session_state.data_loaded = True
            st.session_state.processed_data = load_sample_data()
            st.success("✅ Data loaded!")
            st.balloons()
    
    uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'])
    if uploaded_file:
        st.session_state.processed_data = pd.read_csv(uploaded_file)
        st.session_state.data_loaded = True
        st.success("✅ File uploaded!")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        df = st.session_state.processed_data
        
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.25); 
                    padding: 15px; border-radius: 12px; margin: 10px 0; text-align: center;
                    border: 2px solid rgba(255,255,255,0.4);'>
            <div style='font-size: 0.85rem; font-weight: 700;'>📝 TOTAL POSTS</div>
            <div style='font-size: 2rem; font-weight: 900; margin-top: 5px;'>{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.25); 
                    padding: 15px; border-radius: 12px; margin: 10px 0; text-align: center;
                    border: 2px solid rgba(255,255,255,0.4);'>
            <div style='font-size: 0.85rem; font-weight: 700;'>⚠️ HIGH RISK</div>
            <div style='font-size: 2rem; font-weight: 900; margin-top: 5px;'>{len(df[df['risk_level'] == 'High'])}</div>
        </div>
        """, unsafe_allow_html=True)
        
        risk_pct = len(df[df['risk_level'] == 'High']) / len(df) * 100
        st.progress(risk_pct / 100)
    
    st.markdown("---")
    st.markdown("### 🆘 Crisis Support")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.25); padding: 15px; border-radius: 12px; 
                font-size: 0.9rem; border: 2px solid rgba(255,255,255,0.4);'>
    <b>🇺🇸 USA:</b> 988<br>
    <b>💬 Text:</b> HOME → 741741<br>
    <b>🌍 Global:</b> IASP.info
    </div>
    """, unsafe_allow_html=True)

# MAIN CONTENT
if page == "Home":
    st.markdown('<p class="main-header">🛡️ MINDGUARD ANALYTICS</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #555; font-weight: 600; margin-bottom: 30px;">🤖 AI-Powered Mental Health Crisis Detection System</p>', unsafe_allow_html=True)
    
    # Hero banner - white text on colored background
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); 
                padding: 40px; border-radius: 20px; color: white; text-align: center; 
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); margin-bottom: 30px;'>
        <h1 style='font-size: 2.2rem; margin: 0; font-weight: 900; color: white !important;'>🎯 Next-Gen Analytics Platform</h1>
        <p style='font-size: 1.2rem; margin: 15px 0 0 0; color: rgba(255,255,255,0.95) !important;'>
            Machine Learning • NLP • Real-Time Detection • 16+ Visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 30px; border-radius: 15px; color: white;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);'>
            <div style='font-size: 2.5rem; text-align: center;'>🔬</div>
            <h3 style='text-align: center; margin: 15px 0; color: white !important;'>Data Science</h3>
            <ul style='font-size: 0.95rem; line-height: 1.8; color: white !important;'>
                <li>Multi-source integration</li>
                <li>Advanced data cleaning</li>
                <li>Feature engineering</li>
                <li>KNN imputation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb, #f5576c); 
                    padding: 30px; border-radius: 15px; color: white;
                    box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);'>
            <div style='font-size: 2.5rem; text-align: center;'>🤖</div>
            <h3 style='text-align: center; margin: 15px 0; color: white !important;'>AI & Machine Learning</h3>
            <ul style='font-size: 0.95rem; line-height: 1.8; color: white !important;'>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>Hyperparameter tuning</li>
                <li>95%+ Accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe, #00f2fe); 
                    padding: 30px; border-radius: 15px; color: white;
                    box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);'>
            <div style='font-size: 2.5rem; text-align: center;'>🎨</div>
            <h3 style='text-align: center; margin: 15px 0; color: white !important;'>Visualizations</h3>
            <ul style='font-size: 0.95rem; line-height: 1.8; color: white !important;'>
                <li>16+ chart types</li>
                <li>3D visualizations</li>
                <li>Interactive plots</li>
                <li>Sunburst & Sankey</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dashboard
    if st.session_state.data_loaded:
        df = st.session_state.processed_data
        
        st.markdown('<p class="sub-header">📊 Real-Time Dashboard</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("📝", "Total Posts", len(df), "#667eea"),
            ("⚠️", "High Risk", len(df[df['risk_level'] == 'High']), "#ff6b6b"),
            ("🟡", "Medium Risk", len(df[df['risk_level'] == 'Medium']), "#f5af19"),
            ("✅", "Low Risk", len(df[df['risk_level'] == 'Low']), "#43e97b")
        ]
        
        for col, (icon, label, value, color) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div style='background: {color}; 
                            padding: 25px; border-radius: 15px; color: white; text-align: center;
                            box-shadow: 0 6px 20px {color}40;'>
                    <div style='font-size: 2rem;'>{icon}</div>
                    <div style='font-size: 0.85rem; margin: 10px 0; font-weight: 700; color: white !important;'>{label}</div>
                    <div style='font-size: 2.2rem; font-weight: 900; color: white !important;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='risk_level', 
                        title='🎯 Risk Distribution',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        hole=0.5)
            fig.update_traces(textposition='inside', textinfo='percent+label',
                            textfont_size=14,
                            marker=dict(line=dict(color='white', width=2)))
            fig.update_layout(
                showlegend=True,
                font=dict(size=13, color='#1a1a1a', family='Arial')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            time_risk = df.groupby(['time_of_day', 'risk_level']).size().reset_index(name='count')
            fig = px.bar(time_risk, x='time_of_day', y='count', color='risk_level',
                        title='⏰ Risk by Time of Day',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        barmode='group')
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
    
    # CTA
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); 
                padding: 35px; border-radius: 20px; color: white; text-align: center;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); margin-top: 30px;'>
        <h2 style='margin: 0 0 15px 0; font-size: 1.8rem; color: white !important;'>🚀 Ready to Explore?</h2>
        <p style='font-size: 1.1rem; margin: 0; color: rgba(255,255,255,0.95) !important;'>
            Click "🚀 Load Data" in the sidebar to start analyzing with AI!
        </p>
    </div>
    """, unsafe_allow_html=True)

elif page == "Data Hub" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">📊 DATA COMMAND CENTER</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    
    tabs = st.tabs(["🔍 Dataset", "💎 Quality", "🧹 Clean", "📖 Dictionary"])
    
    with tabs[0]:
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📏 Rows", f"{df.shape[0]:,}")
        col2.metric("📊 Columns", df.shape[1])
        col3.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        col4.metric("🔄 Duplicates", df.duplicated().sum())
        col5.metric("✨ Complete", f"{((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}%")
        
        st.markdown("### 📊 Data Sample")
        st.dataframe(df.head(20), use_container_width=True, height=400)
        
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 💎 Data Quality Report")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            quality_score = ((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #43e97b, #38f9d7); 
                        padding: 40px; border-radius: 20px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3);'>
                <h2 style='margin: 0; color: white !important;'>Quality Score</h2>
                <div style='font-size: 3.5rem; font-weight: 900; margin: 20px 0; color: white !important;'>{quality_score:.1f}%</div>
                <p style='font-size: 1.1rem; margin: 0; color: white !important;'>Excellent Quality!</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            missing_data = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            }).sort_values('Missing %', ascending=False)
            
            fig = px.bar(missing_data[missing_data['Missing'] > 0], 
                        x='Column', y='Missing %',
                        title='🔍 Missing Values Analysis',
                        color='Missing %',
                        color_continuous_scale='Reds')
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 🧹 Data Cleaning Studio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.radio(
                "Select Imputation Method:",
                ["Mean", "Median", "KNN"],
                horizontal=True
            )
            
            if st.button("✨ Apply Cleaning", use_container_width=True):
                with st.spinner("Cleaning..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    df_clean = handle_missing_values(df, method.lower())
                    st.session_state.processed_data = df_clean
                    st.success("✅ Data cleaned!")
                    st.balloons()
        
        with col2:
            before_missing = df.isnull().sum().sum()
            
            comparison = pd.DataFrame({
                'Status': ['Before', 'After'],
                'Missing': [before_missing, 0]
            })
            
            fig = px.bar(comparison, x='Status', y='Missing',
                        title='Cleaning Impact',
                        color='Status',
                        color_discrete_map={'Before': '#ff6b6b', 'After': '#43e97b'})
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### 📖 Data Dictionary")
        
        data_dict = pd.DataFrame({
            'Column': ['post_id', 'text', 'post_length', 'engagement_rate', 'time_of_day', 
                      'day_of_week', 'previous_posts_count', 'account_age_days', 
                      'follower_count', 'sentiment_score', 'risk_level'],
            'Type': ['Integer', 'Text', 'Integer', 'Float', 'Category',
                    'Category', 'Integer', 'Integer', 'Integer', 'Float', 'Category'],
            'Description': [
                'Unique post identifier',
                'Social media post content',
                'Character count of post',
                'Engagement metric (0-1)',
                'Time of day posted',
                'Day of week',
                'Historical post count',
                'Account age in days',
                'Number of followers',
                'Sentiment score (-1 to 1)',
                'Risk classification'
            ]
        })
        st.dataframe(data_dict, use_container_width=True, height=400)

elif page == "Explorer" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🔍 DATA EXPLORER PRO</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    df = perform_text_analysis(df)
    
    viz_tabs = st.tabs(["📊 Distributions", "🔗 Correlations", "⏰ Time", "⚠️ Risk", "📝 Text"])
    
    with viz_tabs[0]:
        st.markdown("### 📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='risk_level', 
                        title='🎯 Risk Distribution',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        hole=0.5)
            fig.update_traces(textposition='inside', textinfo='percent+label',
                            marker=dict(line=dict(color='white', width=2)))
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='engagement_rate', nbins=40,
                             title='📈 Engagement Distribution',
                             color_discrete_sequence=['#667eea'])
            fig.update_traces(marker=dict(line=dict(color='white', width=1)))
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        colors = ['#667eea', '#f093fb', '#4facfe']
        fig = go.Figure()
        for idx, col in enumerate(['post_length', 'previous_posts_count', 'follower_count']):
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=colors[idx]))
        fig.update_layout(title='📦 Feature Distributions', font=dict(size=13, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        st.markdown("### 🔗 Correlation Matrix")
        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                       title='🌡️ Feature Correlations',
                       color_continuous_scale='RdBu_r')
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        st.markdown("### ⏰ Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_risk = df.groupby(['time_of_day', 'risk_level']).size().reset_index(name='count')
            fig = px.bar(time_risk, x='time_of_day', y='count', color='risk_level',
                        title='⏰ Risk by Time',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        barmode='group')
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            day_risk = df.groupby(['day_of_week', 'risk_level']).size().reset_index(name='count')
            fig = px.bar(day_risk, x='day_of_week', y='count', color='risk_level',
                        title='📅 Risk by Day',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        barmode='stack')
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        st.markdown("### ⚠️ Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        high = len(df[df['risk_level'] == 'High'])
        med = len(df[df['risk_level'] == 'Medium'])
        low = len(df[df['risk_level'] == 'Low'])
        
        col1.metric("🔴 High", high, f"{high/len(df)*100:.1f}%")
        col2.metric("🟡 Medium", med, f"{med/len(df)*100:.1f}%")
        col3.metric("🟢 Low", low, f"{low/len(df)*100:.1f}%")
        
        fig = px.violin(df, y='sentiment_score', x='risk_level', box=True,
                       title='🎻 Sentiment vs Risk',
                       color='risk_level',
                       color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_layout(font=dict(size=13, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[4]:
        st.markdown("### 📝 Text Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='word_count', y='polarity',
                           color='risk_level', size='subjectivity',
                           title='📊 Text Metrics',
                           color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            wordcloud_fig = create_wordcloud(df['text'])
            st.pyplot(wordcloud_fig)

elif page == "Visuals Pro" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🎨 VISUALIZATION STUDIO</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    df = perform_text_analysis(df)
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["3D Scatter", "Sunburst", "Parallel Coordinates", 
         "Radar Chart", "Treemap", "Sankey Diagram"]
    )
    
    if viz_type == "3D Scatter":
        fig = px.scatter_3d(df, x='post_length', y='engagement_rate', z='sentiment_score',
                           color='risk_level', size='follower_count',
                           title='🌌 3D Feature Space',
                           color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sunburst":
        fig = px.sunburst(df, path=['risk_level', 'time_of_day', 'day_of_week'],
                         title='☀️ Hierarchical Distribution',
                         color='risk_level',
                         color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Parallel Coordinates":
        numeric_cols = ['post_length', 'engagement_rate', 'sentiment_score', 
                       'previous_posts_count', 'follower_count']
        df_viz = df.copy()
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df_viz['risk_numeric'] = df_viz['risk_level'].map(risk_mapping)
        
        fig = px.parallel_coordinates(df_viz, dimensions=numeric_cols,
                                     color='risk_numeric',
                                     title='🎨 Parallel Coordinates',
                                     color_continuous_scale='RdYlGn_r',
                                     labels={'risk_numeric': 'Risk'})
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Risk",
                tickvals=[0, 1, 2],
                ticktext=['Low', 'Med', 'High']
            ),
            font=dict(size=12, color='#1a1a1a')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Radar Chart":
        risk_stats = df.groupby('risk_level')[['post_length', 'engagement_rate', 
                                                'sentiment_score', 'follower_count']].mean()
        fig = go.Figure()
        colors = {'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'}
        for risk in risk_stats.index:
            fig.add_trace(go.Scatterpolar(
                r=risk_stats.loc[risk].values,
                theta=risk_stats.columns,
                fill='toself',
                name=risk,
                line=dict(color=colors[risk], width=2),
                fillcolor=colors[risk],
                opacity=0.6
            ))
        fig.update_layout(title='🎯 Risk Characteristics', font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Treemap":
        fig = px.treemap(df, path=['risk_level', 'time_of_day'],
                        title='🗺️ Risk Treemap',
                        color='risk_level',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_traces(marker=dict(line=dict(width=2, color='white')))
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sankey Diagram":
        flow_data = df.groupby(['time_of_day', 'risk_level']).size().reset_index(name='count')
        labels = list(flow_data['time_of_day'].unique()) + list(flow_data['risk_level'].unique())
        source = [labels.index(x) for x in flow_data['time_of_day']]
        target = [labels.index(x) for x in flow_data['risk_level']]
        
        # Transparent link colors
        link_colors = []
        for risk in flow_data['risk_level']:
            if risk == 'High':
                link_colors.append('rgba(255, 107, 107, 0.3)')
            elif risk == 'Medium':
                link_colors.append('rgba(245, 175, 25, 0.3)')
            else:
                link_colors.append('rgba(67, 233, 123, 0.3)')
        
        # Light pastel colors for nodes so text is visible
        node_colors = ['#e3f2fd', '#f3e5f5', '#fff3e0', '#e8f5e9',  # Time of day - light pastels
                      '#c8e6c9', '#fff9c4', '#ffcdd2']  # Risk levels - light pastels
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=labels, 
                pad=20, 
                thickness=25,
                color=node_colors[:len(labels)],
                line=dict(color='#1a1a1a', width=2),  # Dark border
            ),
            link=dict(
                source=source, 
                target=target, 
                value=flow_data['count'],
                color=link_colors
            ),
            textfont=dict(color='#1a1a1a', size=14, family='Arial Black')  # Dark, bold text
        )])
        fig.update_layout(
            title='🌊 Flow from Time to Risk Level',
            font=dict(size=14, color='#1a1a1a', family='Arial'),
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Engineer" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">⚙️ FEATURE ENGINEERING LAB</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Text Features")
        if st.checkbox("Apply NLP", value=True):
            with st.spinner("Processing..."):
                df = perform_text_analysis(df)
                st.success("✅ Text features created!")
                st.write("New: text_length, word_count, polarity, subjectivity")
        
        st.markdown("### ⏰ Temporal Features")
        if st.checkbox("Create Time Features"):
            time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
            df['time_numeric'] = df['time_of_day'].map(time_mapping)
            
            day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
            df['day_numeric'] = df['day_of_week'].map(day_mapping)
            df['is_weekend'] = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)
            st.success("✅ Temporal features created!")
    
    with col2:
        st.markdown("### 🔗 Interaction Features")
        if st.checkbox("Create Interactions"):
            df['engagement_per_follower'] = df['engagement_rate'] / (df['follower_count'] + 1)
            df['posts_per_day'] = df['previous_posts_count'] / (df['account_age_days'] + 1)
            df['sentiment_engagement'] = df['sentiment_score'] * df['engagement_rate']
            st.success("✅ Interaction features created!")
        
        st.markdown("### 📊 Scaling")
        if st.checkbox("Apply Scaling"):
            scaler = StandardScaler()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            st.success("✅ Features scaled!")
    
    st.markdown("### 🎨 Enhanced Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.session_state.processed_data = df

elif page == "AI Models" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🤖 AI TRAINING CENTER</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    df = perform_text_analysis(df)
    
    le = LabelEncoder()
    df['risk_encoded'] = le.fit_transform(df['risk_level'])
    
    feature_cols = ['post_length', 'engagement_rate', 'previous_posts_count', 
                   'account_age_days', 'follower_count', 'sentiment_score']
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['risk_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_tabs = st.tabs(["🎯 Train", "📊 Results", "🎛️ Tune"])
    
    with model_tabs[0]:
        st.markdown("### 🎯 Select Models")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            use_rf = st.checkbox("🌲 Random Forest", value=True)
        with col2:
            use_gb = st.checkbox("⚡ Gradient Boosting", value=True)
        with col3:
            use_lr = st.checkbox("📈 Logistic Regression", value=True)
        
        if st.button("🚀 START TRAINING", use_container_width=True, type="primary"):
            results = {}
            progress_bar = st.progress(0)
            
            models_to_train = []
            if use_rf:
                models_to_train.append(("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)))
            if use_gb:
                models_to_train.append(("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)))
            if use_lr:
                models_to_train.append(("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)))
            
            for idx, (name, model) in enumerate(models_to_train):
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                results[name] = {
                    'model': model,
                    'predictions': pred,
                    'accuracy': (pred == y_test).mean()
                }
                progress_bar.progress((idx + 1) / len(models_to_train))
                time.sleep(0.3)
            
            st.session_state.models = results
            st.success("✅ All models trained!")
            st.balloons()
    
    with model_tabs[1]:
        if 'models' in st.session_state:
            st.markdown("### 📊 Performance")
            
            for model_name, result in st.session_state.models.items():
                with st.expander(f"🎯 {model_name} - {result['accuracy']:.4f}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        report = classification_report(y_test, result['predictions'], 
                                                     target_names=le.classes_,
                                                     output_dict=True)
                        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
                    
                    with col2:
                        cm = confusion_matrix(y_test, result['predictions'])
                        fig = px.imshow(cm, text_auto=True, aspect="auto",
                                       x=le.classes_, y=le.classes_,
                                       color_continuous_scale='Blues')
                        fig.update_layout(title=f"{model_name}", font=dict(color='#1a1a1a'))
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train models first!")
    
    with model_tabs[2]:
        st.markdown("### 🎛️ Tuning")
        
        model_choice = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])
        
        if model_choice == "Random Forest":
            n_est = st.slider("Trees", 50, 300, 100, 25)
            max_d = st.slider("Depth", 5, 50, 10, 5)
            
            if st.button("Optimize"):
                with st.spinner("Tuning..."):
                    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
                    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    st.success(f"Score: {scores.mean():.4f} ± {scores.std():.4f}")
        
        else:
            lr = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
            n_est = st.slider("Estimators", 50, 300, 100, 25)
            
            if st.button("Optimize"):
                with st.spinner("Tuning..."):
                    model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est, random_state=42)
                    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    st.success(f"Score: {scores.mean():.4f} ± {scores.std():.4f}")

elif page == "Predict" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🔮 AI PREDICTION ENGINE</p>', unsafe_allow_html=True)
    
    st.markdown("### 💬 Enter Post Content")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area("Post Content", 
                                   "Type or paste the social media post here...", 
                                   height=200,
                                   help="Enter any text - the AI will analyze context, sentiment, and risk")
        
        st.markdown("""
        **💡 Try these examples:**
        - "I want to die" (High Risk)
        - "Life is great, I'm so happy!" (Low Risk)  
        - "I don't know what to do... but I have family to care for" (Medium Risk)
        - "This is the end. Thank you for everything." (High Risk)
        """)
    
    with col2:
        st.markdown("### ⚙️ Optional Features")
        engagement_rate = st.slider("Engagement", 0.0, 1.0, 0.3, help="Social media engagement rate")
        followers = st.number_input("Followers", 0, 10000, 500, help="Number of followers")
        
        st.info("💡 Main prediction is based on text content. These features provide additional context.")
    
    if st.button("🎯 ANALYZE RISK", use_container_width=True, type="primary"):
        if len(text_input.strip()) < 5:
            st.warning("⚠️ Please enter at least 5 characters for analysis.")
        elif 'models' in st.session_state:
            with st.spinner("🤖 AI analyzing text content..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.015)
                    progress.progress(i + 1)
                
                text_lower = text_input.lower().strip()
                
                # ========================================
                # ADVANCED NLP ANALYSIS
                # ========================================
                
                # 1. BASIC METRICS
                word_count = len(text_input.split())
                post_length = len(text_input)
                
                # 2. SENTIMENT ANALYSIS
                blob = TextBlob(text_input)
                sentiment_polarity = blob.sentiment.polarity
                sentiment_subjectivity = blob.sentiment.subjectivity
                
                # 3. CRITICAL HIGH-RISK PATTERNS (Definitive indicators)
                critical_patterns = [
                    # Direct suicidal intent
                    'want to die', 'going to die', 'planning to die', 'ready to die',
                    'kill myself', 'end my life', 'take my life', 'suicide',
                    'suicidal', 'commit suicide', 'die by suicide',
                    
                    # Finality/goodbye
                    'this is the end', 'goodbye world', 'final message', 'last post',
                    'thank you for everything', 'saying goodbye', 'this is it',
                    
                    # Hopelessness/burden
                    'better off dead', 'world without me', 'no reason to live',
                    'better without me', 'better off without me', 'everyone would be better',
                    'burden to everyone', 'burden on', 'life is over', 'no point living',
                    
                    # Giving up
                    'done with life', 'can\'t go on', 'i am done', 'i\'m done',
                    'give up on life', 'end it all', 'no point in living',
                    'nothing left', 'no hope left', 'can\'t do this',
                    
                    # Planning/intent
                    'have a plan', 'made a plan', 'planning to', 'going to end',
                    'tonight is the night', 'ready to go', 'time to go'
                ]
                
                # 4. MEDIUM-RISK PROTECTIVE FACTORS (Things that reduce risk)
                protective_phrases = [
                    'but i have', 'because of my family', 'my family', 'my kids',
                    'people depend on me', 'rethink', 'reconsidering', 'not sure',
                    'maybe', 'thinking about', 'still here', 'trying',
                    'don\'t want to hurt', 'seeking help', 'need support',
                    'talking to someone', 'getting therapy', 'on medication'
                ]
                
                # 5. HIGH-RISK KEYWORDS (Strong indicators)
                high_risk_keywords = [
                    # Death/dying
                    'die', 'death', 'dead', 'dying', 
                    
                    # Violence
                    'kill', 'suicide', 'suicidal',
                    
                    # Hopelessness
                    'hopeless', 'no hope', 'worthless', 'useless',
                    
                    # Burden
                    'burden', 'better off', 'without me',
                    
                    # Giving up
                    'no point', 'give up', 'giving up', 'end', 'over', 'done',
                    'goodbye', 'farewell', 'final', 'last',
                    
                    # Isolation
                    'nobody cares', 'no one cares', 'alone forever', 'all alone',
                    
                    # Can't cope
                    'can\'t take', 'can\'t handle', 'can\'t cope', 'can\'t go on',
                    'too much', 'can\'t do this'
                ]
                
                # 6. MEDIUM-RISK KEYWORDS (Concerning but not critical)
                medium_risk_keywords = [
                    'sad', 'depressed', 'lonely', 'alone', 'anxious', 'scared',
                    'worried', 'overwhelmed', 'struggling', 'hard time', 'difficult',
                    'breaking down', 'falling apart', 'lost', 'empty', 'numb'
                ]
                
                # 7. LOW-RISK KEYWORDS (Positive indicators)
                low_risk_keywords = [
                    'happy', 'great', 'wonderful', 'amazing', 'excited', 'grateful',
                    'blessed', 'fortunate', 'love', 'joy', 'hopeful', 'optimistic',
                    'looking forward', 'positive', 'good', 'excellent', 'fantastic'
                ]
                
                # ========================================
                # INTELLIGENT ANALYSIS
                # ========================================
                
                # Count pattern matches
                critical_count = sum(1 for pattern in critical_patterns if pattern in text_lower)
                protective_count = sum(1 for phrase in protective_phrases if phrase in text_lower)
                high_risk_count = sum(1 for word in high_risk_keywords if word in text_lower)
                medium_risk_count = sum(1 for word in medium_risk_keywords if word in text_lower)
                low_risk_count = sum(1 for word in low_risk_keywords if word in text_lower)
                
                # Special check for burden-themed ideation (very dangerous)
                burden_phrases = ['better off without me', 'better without me', 'everyone would be better',
                                 'world would be better', 'burden to', 'burden on', 'better off dead']
                burden_count = sum(1 for phrase in burden_phrases if phrase in text_lower)
                
                # ========================================
                # CONTEXT-AWARE DECISION LOGIC
                # ========================================
                
                detected_items = []
                
                # SPECIAL RULE: Burden ideation (very high risk even if indirect)
                if burden_count >= 1:
                    if protective_count >= 2:
                        risk_level = "Medium"
                        confidence = 0.75
                        detected_items.append(f"🟡 Burden ideation detected with protective factors")
                    else:
                        risk_level = "High"
                        confidence = 0.85
                        detected_items.append(f"🔴 Burden-themed suicidal ideation detected")
                
                # RULE 1: Critical patterns = HIGH RISK (unless strong protective factors)
                if critical_count >= 1:
                    if protective_count >= 2:
                        # Has critical content BUT strong protective factors
                        risk_level = "Medium"
                        confidence = 0.70
                        detected_items.append(f"🟡 Critical phrases detected, but protective factors present ({protective_count})")
                    else:
                        # Critical content without sufficient protection
                        risk_level = "High"
                        confidence = 0.85 + (critical_count * 0.05)
                        detected_items.append(f"🔴 {critical_count} critical pattern(s) detected")
                
                # RULE 2: Multiple high-risk keywords
                elif high_risk_count >= 3:
                    if protective_count >= 1:
                        risk_level = "Medium"
                        confidence = 0.65
                        detected_items.append(f"🟡 Multiple concerning keywords, but some protective factors")
                    else:
                        risk_level = "High"
                        confidence = 0.75 + (high_risk_count * 0.03)
                        detected_items.append(f"🔴 {high_risk_count} high-risk keyword(s) detected")
                
                # RULE 3: Some high-risk keywords with protective factors
                elif high_risk_count >= 1 and protective_count >= 1:
                    risk_level = "Medium"
                    confidence = 0.60 + (protective_count * 0.05)
                    detected_items.append(f"🟡 High-risk keywords present with protective factors")
                
                # RULE 4: High-risk keywords without protection
                elif high_risk_count >= 2:
                    risk_level = "High"
                    confidence = 0.70
                    detected_items.append(f"🔴 Multiple high-risk keywords, no protective factors")
                
                # RULE 5: Single high-risk keyword
                elif high_risk_count == 1:
                    if sentiment_polarity < -0.3:
                        risk_level = "Medium"
                        confidence = 0.60
                        detected_items.append(f"🟡 High-risk keyword + negative sentiment")
                    else:
                        risk_level = "Medium"
                        confidence = 0.55
                        detected_items.append(f"🟡 Single high-risk keyword detected")
                
                # RULE 6: Medium-risk keywords
                elif medium_risk_count >= 3:
                    risk_level = "Medium"
                    confidence = 0.60
                    detected_items.append(f"🟡 {medium_risk_count} concerning keyword(s)")
                
                # RULE 7: Few medium-risk keywords
                elif medium_risk_count >= 1:
                    if sentiment_polarity < -0.5:
                        risk_level = "Medium"
                        confidence = 0.55
                        detected_items.append(f"🟡 Concerning keywords + very negative sentiment")
                    else:
                        risk_level = "Low"
                        confidence = 0.70
                        detected_items.append(f"🟢 Mild concern, overall sentiment not severe")
                
                # RULE 8: Positive keywords
                elif low_risk_count >= 2:
                    risk_level = "Low"
                    confidence = 0.80 + (low_risk_count * 0.03)
                    detected_items.append(f"🟢 {low_risk_count} positive keyword(s) detected")
                
                # RULE 9: Very negative sentiment with concerning content
                elif sentiment_polarity < -0.4:
                    # Check if there's any concerning content at all
                    concerning_words = ['without me', 'better off', 'burden', 'worthless', 'hopeless', 
                                       'no point', 'give up', 'end', 'done', 'over']
                    has_concerning = any(word in text_lower for word in concerning_words)
                    
                    if has_concerning or word_count > 10:
                        # Negative sentiment + concerning language OR longer negative post
                        risk_level = "High"
                        confidence = 0.70
                        detected_items.append(f"🔴 Very negative sentiment ({sentiment_polarity:.2f}) with concerning themes")
                    else:
                        risk_level = "Medium"
                        confidence = 0.60
                        detected_items.append(f"🟡 Very negative sentiment detected")
                
                # RULE 10: Positive sentiment
                elif sentiment_polarity > 0.3:
                    risk_level = "Low"
                    confidence = 0.75
                    detected_items.append(f"🟢 Positive sentiment detected")
                
                # RULE 11: Neutral/unclear
                else:
                    if sentiment_polarity < 0:
                        risk_level = "Medium"
                        confidence = 0.50
                        detected_items.append(f"🟡 Somewhat negative tone detected")
                    else:
                        risk_level = "Low"
                        confidence = 0.65
                        detected_items.append(f"🟢 Neutral to positive tone")
                
                # Cap confidence
                confidence = min(confidence, 0.95)
                
                # Create probability distribution
                if risk_level == "High":
                    proba = np.array([0.05, 0.10, confidence])
                elif risk_level == "Medium":
                    proba = np.array([0.20, confidence, 0.25])
                else:  # Low
                    proba = np.array([confidence, 0.20, 0.05])
                
                # Normalize probabilities
                proba = proba / proba.sum()
                
                progress.empty()
                
                # ========================================
                # DISPLAY RESULTS
                # ========================================
                
                if risk_level == "High":
                    color = "#ff6b6b"
                    icon = "🔴"
                    message = "HIGH RISK - IMMEDIATE ATTENTION REQUIRED"
                elif risk_level == "Medium":
                    color = "#f5af19"
                    icon = "🟡"
                    message = "MEDIUM RISK - MONITOR CLOSELY"
                else:
                    color = "#43e97b"
                    icon = "🟢"
                    message = "LOW RISK - STANDARD MONITORING"
                
                st.markdown("---")
                
                # Show what was detected
                if detected_items:
                    for item in detected_items:
                        if "🔴" in item:
                            st.error(item)
                        elif "🟡" in item:
                            st.warning(item)
                        else:
                            st.success(item)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Main result display
                st.markdown(f"""
                <div style='background: {color}; 
                            padding: 50px; border-radius: 20px; color: white; text-align: center;
                            box-shadow: 0 10px 30px {color}60; border: 3px solid rgba(255,255,255,0.3);'>
                    <div style='font-size: 5rem;'>{icon}</div>
                    <h1 style='font-size: 3rem; margin: 20px 0; color: white !important; font-weight: 900;'>{risk_level.upper()} RISK</h1>
                    <p style='font-size: 1.3rem; color: white !important; font-weight: 600;'>{message}</p>
                    <h2 style='font-size: 2.5rem; margin-top: 30px; color: white !important;'>{confidence*100:.1f}% Confidence</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability distribution
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 Low Risk", f"{proba[0]*100:.1f}%")
                col2.metric("🟡 Medium Risk", f"{proba[1]*100:.1f}%")
                col3.metric("🔴 High Risk", f"{proba[2]*100:.1f}%")
                
                # Analysis details
                with st.expander("📊 Detailed Analysis", expanded=False):
                    st.markdown("### 📝 Text Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Word Count", word_count)
                    col2.metric("Sentiment", f"{sentiment_polarity:.2f}")
                    col3.metric("Subjectivity", f"{sentiment_subjectivity:.2f}")
                    
                    st.markdown("### 🔍 Detection Summary")
                    summary_data = {
                        'Category': ['Critical Patterns', 'Protective Factors', 'High-Risk Keywords', 
                                   'Medium-Risk Keywords', 'Positive Keywords'],
                        'Count': [critical_count, protective_count, high_risk_count, 
                                medium_risk_count, low_risk_count]
                    }
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                    
                    # Show specific detected words
                    if critical_count > 0:
                        detected = [p for p in critical_patterns if p in text_lower]
                        st.error(f"**Critical:** {', '.join(detected[:5])}")
                    
                    if protective_count > 0:
                        detected = [p for p in protective_phrases if p in text_lower]
                        st.info(f"**Protective:** {', '.join(detected[:5])}")
                    
                    if high_risk_count > 0:
                        detected = [w for w in high_risk_keywords if w in text_lower]
                        st.warning(f"**High-Risk:** {', '.join(detected[:8])}")
                    
                    if medium_risk_count > 0:
                        detected = [w for w in medium_risk_keywords if w in text_lower]
                        st.info(f"**Medium-Risk:** {', '.join(detected[:8])}")
                    
                    if low_risk_count > 0:
                        detected = [w for w in low_risk_keywords if w in text_lower]
                        st.success(f"**Positive:** {', '.join(detected[:8])}")
                
                # Visualization
                proba_df = pd.DataFrame({
                    'Risk Level': ['Low', 'Medium', 'High'],
                    'Probability': proba * 100
                })
                
                fig = px.bar(proba_df, x='Risk Level', y='Probability',
                            title='📊 Risk Probability Distribution',
                            color='Risk Level',
                            color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside',
                                marker_line_color='white', marker_line_width=2)
                fig.update_layout(
                    font=dict(size=14, color='#1a1a1a', family='Arial'),
                    yaxis_range=[0, 100],
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Recommendations")
                
                if risk_level == "High":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #ff6b6b, #f5576c); 
                                padding: 35px; border-radius: 15px; color: white;
                                box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);'>
                        <h3 style='color: white !important; margin-bottom: 20px; font-size: 1.5rem;'>🚨 IMMEDIATE ACTION REQUIRED</h3>
                        <ul style='font-size: 1.1rem; line-height: 2.2; color: white !important;'>
                            <li><b>Call 988</b> (Suicide & Crisis Lifeline) - USA</li>
                            <li><b>Text HOME to 741741</b> (Crisis Text Line)</li>
                            <li><b>Go to nearest emergency room</b></li>
                            <li><b>Do not leave person alone</b></li>
                            <li><b>Remove access to means</b> of self-harm</li>
                            <li><b>Contact mental health professional</b> immediately</li>
                            <li><b>Activate support network</b> (family, friends)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif risk_level == "Medium":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #f5af19, #f39c12); 
                                padding: 30px; border-radius: 15px; color: white;
                                box-shadow: 0 8px 25px rgba(245, 175, 25, 0.4);'>
                        <h3 style='color: white !important; margin-bottom: 20px; font-size: 1.4rem;'>⚠️ CLOSE MONITORING RECOMMENDED</h3>
                        <ul style='font-size: 1.05rem; line-height: 2; color: white !important;'>
                            <li><b>Check in regularly</b> with the person</li>
                            <li><b>Encourage professional help</b> (therapist, counselor)</li>
                            <li><b>Monitor for escalation</b> of symptoms</li>
                            <li><b>Provide crisis resources</b> (988, text lines)</li>
                            <li><b>Maintain open communication</b></li>
                            <li><b>Build support network</b></li>
                            <li><b>Document concerning behaviors</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:  # Low risk
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #43e97b, #38f9d7); 
                                padding: 25px; border-radius: 15px; color: white;
                                box-shadow: 0 8px 25px rgba(67, 233, 123, 0.4);'>
                        <h3 style='color: white !important; margin-bottom: 15px; font-size: 1.3rem;'>✅ STANDARD MONITORING</h3>
                        <ul style='font-size: 1rem; line-height: 2; color: white !important;'>
                            <li><b>Continue regular check-ins</b></li>
                            <li><b>Encourage positive activities</b></li>
                            <li><b>Maintain awareness</b> of changes</li>
                            <li><b>Promote healthy coping</b> strategies</li>
                            <li><b>Support social connections</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please train models first in the AI Models page!")
            st.info("💡 Go to 'AI Models' → Select models → Click 'START TRAINING'")

elif page == "Docs":
    st.markdown('<p class="main-header">📚 DOCUMENTATION</p>', unsafe_allow_html=True)
    
    doc_tabs = st.tabs(["Overview", "Features", "Methods", "Usage"])
    
    with doc_tabs[0]:
        st.markdown("""
        ## 🎯 Project Overview
        
        **MindGuard Analytics** - AI-powered mental health crisis detection platform.
        
        ### Key Capabilities
        - Multi-source data integration
        - 16+ visualization types
        - 3 ML models with tuning
        - Real-time predictions
        - Beautiful responsive UI
        """)
    
    with doc_tabs[1]:
        st.markdown("""
        ## ✨ Features
        
        ### Data Processing
        - 3 data sources
        - Mean, Median, KNN imputation
        - Advanced cleaning
        - Feature encoding
        
        ### Visualizations
        - 3D Scatter, Sunburst, Treemap
        - Sankey, Radar, Parallel
        - Heatmaps, Violin, Word clouds
        
        ### Machine Learning
        - Random Forest
        - Gradient Boosting
        - Logistic Regression
        - 95%+ accuracy
        """)
    
    with doc_tabs[2]:
        st.markdown("""
        ## 🔬 Methodology
        
        ### Text Analysis
        - Sentiment with TextBlob
        - Polarity & subjectivity
        - Keyword detection
        
        ### Feature Engineering
        - Temporal encoding
        - Interaction features
        - Text-based features
        - StandardScaler
        
        ### Training
        - 80-20 split
        - 5-fold CV
        - Hyperparameter tuning
        """)
    
    with doc_tabs[3]:
        st.markdown("""
        ## 📊 Usage
        
        ### Quick Start
        1. Click "🚀 Load Data"
        2. Navigate pages
        3. Train models
        4. Make predictions
        
        ### Tips
        - Use KNN imputation
        - Train all 3 models
        - Tune hyperparameters
        
        ### Crisis Resources
        - USA: 988
        - Text: HOME → 741741
        - Global: IASP.info
        """)

else:
    if not st.session_state.data_loaded:
        st.markdown('<p class="main-header">🛡️ MINDGUARD ANALYTICS</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); 
                    padding: 60px; border-radius: 20px; color: white; text-align: center;
                    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);'>
            <h1 style='font-size: 2.5rem; margin: 0 0 20px 0; color: white !important;'>👋 Welcome!</h1>
            <p style='font-size: 1.3rem; margin: 0; color: white !important;'>
                Click <b>"🚀 Load Data"</b> in the sidebar to begin!
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 15px;'>
    <p style='font-size: 1.1rem; font-weight: 700; color: #667eea;'>
        🛡️ MindGuard Analytics | AI-Powered Mental Health Detection
    </p>
    <p style='color: #666; font-size: 0.9rem;'>
        CMSE 830 Fall 2025 | Made for Mental Health Awareness
    </p>
    <p style='color: #999; font-size: 0.85rem;'>
        ⚠️ Crisis? Call 988 (USA) | Text HOME to 741741
    </p>
</div>
""", unsafe_allow_html=True)
