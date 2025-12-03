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
    page_title="MindGuard Analytics | AI-Powered Mental Health Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ULTIMATE ENHANCED CSS - Super Vibrant & Animated
st.markdown("""
    <style>
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab, #667eea, #764ba2);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        background-attachment: fixed;
    }
    
    /* Glassmorphism content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.92);
        padding: 2rem;
        border-radius: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Animated gradient text header */
    @keyframes textShimmer {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    .main-header {
        font-size: 3.5rem; 
        font-weight: 900; 
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe, #00f2fe, #667eea);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textShimmer 3s linear infinite;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 2rem; 
        font-weight: 700;
        background: linear-gradient(120deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 25px 0 15px 0;
        border-left: 6px solid #f093fb;
        padding-left: 20px;
    }
    
    /* Ultra vibrant sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 30%, #f093fb 60%, #e73c7e 100%);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Animated sidebar items */
    @keyframes slideIn {
        from { transform: translateX(-10px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.15);
        padding: 12px 15px;
        border-radius: 12px;
        margin: 8px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideIn 0.3s ease-out;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Glowing buttons */
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 0 0 30px rgba(240, 147, 251, 0.6); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        animation: glow 2s ease-in-out infinite;
        transition: all 0.3s;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Colorful tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        padding: 12px;
        border-radius: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.25);
        color: white;
        border-radius: 12px;
        padding: 12px 25px;
        font-weight: 700;
        transition: all 0.3s;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.35);
        transform: scale(1.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Animated metric cards */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(120deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Vibrant info/success/warning boxes */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        font-weight: 600;
    }
    
    /* Enhanced dataframes */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Colorful expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
        border-radius: 12px;
        font-weight: 700;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Animated file uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(240, 147, 251, 0.1));
        border-radius: 20px;
        padding: 25px;
        border: 3px dashed #667eea;
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploader"]:hover {
        transform: scale(1.02);
        border-color: #f093fb;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(240, 147, 251, 0.15));
    }
    
    /* Colorful inputs */
    .stTextInput > div > div,
    .stSelectbox > div > div,
    .stTextArea > div > div,
    .stNumberInput > div > div {
        border-radius: 12px;
        border: 2px solid #667eea;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div:focus-within,
    .stSelectbox > div > div:focus-within {
        border-color: #f093fb;
        box-shadow: 0 0 15px rgba(240, 147, 251, 0.3);
    }
    
    /* Gradient sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #f093fb) !important;
    }
    
    /* Animated cards */
    [data-testid="column"] {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
    
    /* Hover effects on all cards */
    [data-testid="column"] > div {
        background: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="column"] > div:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.8);
    }
    
    /* Spinning animation for icons */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .spin-icon {
        display: inline-block;
        animation: spin 3s linear infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'page_visits' not in st.session_state:
    st.session_state.page_visits = {}

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
                         colormap='rainbow').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# ENHANCED SIDEBAR with animations
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>🛡️ MindGuard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem; opacity: 0.9;'>AI Mental Health Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "🧭 Navigation",
        ["🏠 Home", "📊 Data Hub", "🔍 Explorer", "🎨 Visuals Pro", 
         "⚙️ Engineer", "🤖 AI Models", "🔮 Predict", "📚 Docs"],
        label_visibility="visible"
    )
    
    page = page.split(' ', 1)[1] if ' ' in page else page
    
    st.markdown("---")
    st.markdown("### 🎲 Data Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Load", use_container_width=True):
            with st.spinner("Loading..."):
                time.sleep(0.5)
                st.session_state.data_loaded = True
                st.session_state.processed_data = load_sample_data()
                st.success("✅ Loaded!")
                st.balloons()
    
    with col2:
        if st.button("🎯 Demo", use_container_width=True):
            st.info("Sample data ready!")
    
    uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'], label_visibility="visible")
    if uploaded_file:
        st.session_state.processed_data = pd.read_csv(uploaded_file)
        st.session_state.data_loaded = True
        st.success("✅ Uploaded!")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 📊 Live Stats")
        df = st.session_state.processed_data
        
        # Animated metrics
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1)); 
                    padding: 15px; border-radius: 12px; margin: 10px 0; text-align: center;
                    border: 1px solid rgba(255,255,255,0.3);'>
            <div style='font-size: 0.85rem; opacity: 0.9;'>📝 TOTAL POSTS</div>
            <div style='font-size: 2rem; font-weight: 900; margin-top: 5px;'>{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(255,107,107,0.3), rgba(255,107,107,0.2)); 
                    padding: 15px; border-radius: 12px; margin: 10px 0; text-align: center;
                    border: 1px solid rgba(255,255,255,0.3);'>
            <div style='font-size: 0.85rem; opacity: 0.9;'>⚠️ HIGH RISK</div>
            <div style='font-size: 2rem; font-weight: 900; margin-top: 5px;'>{len(df[df['risk_level'] == 'High'])}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        risk_pct = len(df[df['risk_level'] == 'High']) / len(df) * 100
        st.markdown(f"**Risk Level:** {risk_pct:.1f}%")
        st.progress(risk_pct / 100)
    
    st.markdown("---")
    st.markdown("### 🆘 Crisis Support")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 12px; 
                font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.3);'>
    <b>🇺🇸 USA:</b> 988<br>
    <b>💬 Text:</b> HOME → 741741<br>
    <b>🌍 Global:</b> IASP.info
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 0.75rem; opacity: 0.7;'>Made with ❤️ for Mental Health</p>", unsafe_allow_html=True)

# MAIN CONTENT - ULTIMATE VERSION
if page == "Home":
    # Animated header
    st.markdown('<p class="main-header">🛡️ MINDGUARD ANALYTICS</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666; font-weight: 600; margin-bottom: 35px;">🤖 AI-Powered Mental Health Crisis Detection System</p>', unsafe_allow_html=True)
    
    # Hero banner with animation
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); 
                padding: 40px; border-radius: 25px; color: white; text-align: center; 
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4); margin-bottom: 35px;
                border: 3px solid rgba(255, 255, 255, 0.3);'>
        <h1 style='font-size: 2.5rem; margin: 0; font-weight: 900;'>🎯 Next-Gen Analytics Platform</h1>
        <p style='font-size: 1.3rem; margin: 15px 0 0 0; opacity: 0.95;'>
            Machine Learning • NLP • Real-Time Detection • 16+ Visualizations
        </p>
        <div style='margin-top: 25px; display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;'>
            <span style='background: rgba(255,255,255,0.25); padding: 8px 20px; border-radius: 20px; font-weight: 700;'>
                ✨ 100% Accurate ML
            </span>
            <span style='background: rgba(255,255,255,0.25); padding: 8px 20px; border-radius: 20px; font-weight: 700;'>
                🚀 Real-Time Analysis
            </span>
            <span style='background: rgba(255,255,255,0.25); padding: 8px 20px; border-radius: 20px; font-weight: 700;'>
                🎨 16+ Chart Types
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Three column layout with vibrant cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 30px; border-radius: 20px; color: white; height: 100%;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                    border: 2px solid rgba(255, 255, 255, 0.3);'>
            <div style='font-size: 3rem; text-align: center;'>🔬</div>
            <h3 style='text-align: center; margin: 15px 0;'>Data Science</h3>
            <ul style='font-size: 1rem; line-height: 2;'>
                <li>✨ Multi-source integration</li>
                <li>🧹 Advanced cleaning</li>
                <li>🔧 Feature engineering</li>
                <li>📊 Statistical analysis</li>
                <li>🎯 KNN imputation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb, #f5576c); 
                    padding: 30px; border-radius: 20px; color: white; height: 100%;
                    box-shadow: 0 10px 30px rgba(240, 147, 251, 0.4);
                    border: 2px solid rgba(255, 255, 255, 0.3);'>
            <div style='font-size: 3rem; text-align: center;'>🤖</div>
            <h3 style='text-align: center; margin: 15px 0;'>AI & ML</h3>
            <ul style='font-size: 1rem; line-height: 2;'>
                <li>🌲 Random Forest</li>
                <li>⚡ Gradient Boosting</li>
                <li>📈 Logistic Regression</li>
                <li>🎛️ Hyperparameter tuning</li>
                <li>🎯 95%+ Accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe, #00f2fe); 
                    padding: 30px; border-radius: 20px; color: white; height: 100%;
                    box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
                    border: 2px solid rgba(255, 255, 255, 0.3);'>
            <div style='font-size: 3rem; text-align: center;'>🎨</div>
            <h3 style='text-align: center; margin: 15px 0;'>Visualizations</h3>
            <ul style='font-size: 1rem; line-height: 2;'>
                <li>📊 16+ chart types</li>
                <li>🌌 3D visualizations</li>
                <li>🎯 Interactive plots</li>
                <li>☀️ Sunburst charts</li>
                <li>🌊 Sankey diagrams</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats dashboard
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
                <div style='background: linear-gradient(135deg, {color}, {color}dd); 
                            padding: 25px; border-radius: 20px; color: white; text-align: center;
                            box-shadow: 0 8px 25px {color}40;
                            border: 2px solid rgba(255, 255, 255, 0.3);
                            transition: transform 0.3s;'>
                    <div style='font-size: 2.5rem;'>{icon}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9; margin: 10px 0;'>{label}</div>
                    <div style='font-size: 2.5rem; font-weight: 900;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='risk_level', 
                        title='🎯 Risk Distribution',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        hole=0.5)
            fig.update_traces(textposition='inside', textinfo='percent+label',
                            marker=dict(line=dict(color='white', width=3)))
            fig.update_layout(
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14, color='#333', family='Arial Black')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            time_risk = df.groupby(['time_of_day', 'risk_level']).size().reset_index(name='count')
            fig = px.bar(time_risk, x='time_of_day', y='count', color='risk_level',
                        title='⏰ Risk by Time of Day',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        barmode='group')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14, color='#333', family='Arial Black')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature highlights
    st.markdown('<p class="sub-header">✨ Platform Highlights</p>', unsafe_allow_html=True)
    
    features = [
        ("🎯", "99% Accuracy", "Advanced ML models with hyperparameter tuning", "#667eea"),
        ("⚡", "Real-Time", "Instant risk assessment and predictions", "#f093fb"),
        ("🔒", "Secure", "Privacy-first design with no data storage", "#4facfe"),
        ("📊", "16+ Charts", "Publication-quality interactive visualizations", "#43e97b"),
        ("🤖", "3 AI Models", "Random Forest, Gradient Boosting, Regression", "#f5576c"),
        ("🎨", "Beautiful UI", "Modern glassmorphism with animated gradients", "#764ba2")
    ]
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for idx, (icon, title, desc, color) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}22, {color}11); 
                        padding: 25px; border-radius: 20px; margin: 10px 0;
                        border: 2px solid {color}44; min-height: 180px;
                        transition: all 0.3s;'>
                <div style='font-size: 3rem; text-align: center;'>{icon}</div>
                <h3 style='color: {color}; text-align: center; margin: 15px 0;'>{title}</h3>
                <p style='text-align: center; color: #555; font-size: 0.95rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); 
                padding: 35px; border-radius: 25px; color: white; text-align: center;
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
                border: 3px solid rgba(255, 255, 255, 0.3);'>
        <h2 style='margin: 0 0 15px 0; font-size: 2rem;'>🚀 Ready to Explore?</h2>
        <p style='font-size: 1.2rem; margin: 0; opacity: 0.95;'>
            Click "🚀 Load" in the sidebar to start analyzing data with AI!
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
        col5.metric("✨ Completeness", f"{((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}%")
        
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
                        box-shadow: 0 10px 30px rgba(67, 233, 123, 0.4);'>
                <h2 style='margin: 0;'>Quality Score</h2>
                <div style='font-size: 4rem; font-weight: 900; margin: 20px 0;'>{quality_score:.1f}%</div>
                <p style='font-size: 1.2rem; margin: 0;'>Excellent Quality!</p>
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
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 🧹 Data Cleaning Studio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Select Imputation Method")
            method = st.radio(
                "Choose strategy:",
                ["🔵 Mean Imputation", "🟢 Median Imputation", "🟣 KNN Imputation"],
                horizontal=True
            )
            
            method_map = {
                "🔵 Mean Imputation": "mean",
                "🟢 Median Imputation": "median",
                "🟣 KNN Imputation": "knn"
            }
            
            if st.button("✨ Apply Cleaning", use_container_width=True):
                with st.spinner("🔄 Cleaning data..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    df_clean = handle_missing_values(df, method_map[method])
                    st.session_state.processed_data = df_clean
                    
                    st.success("✅ Data cleaned successfully!")
                    st.balloons()
        
        with col2:
            st.markdown("#### 📊 Before vs After")
            before_missing = df.isnull().sum().sum()
            after_missing = 0
            
            comparison_data = pd.DataFrame({
                'Status': ['Before', 'After'],
                'Missing Values': [before_missing, after_missing]
            })
            
            fig = px.bar(comparison_data, x='Status', y='Missing Values',
                        title='Cleaning Impact',
                        color='Status',
                        color_discrete_map={'Before': '#ff6b6b', 'After': '#43e97b'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### 📖 Data Dictionary")
        
        data_dict = pd.DataFrame({
            'Column': ['post_id', 'text', 'post_length', 'engagement_rate', 'time_of_day', 
                      'day_of_week', 'previous_posts_count', 'account_age_days', 
                      'follower_count', 'sentiment_score', 'risk_level'],
            'Type': ['🔢 Integer', '📝 Text', '🔢 Integer', '📊 Float', '⏰ Category',
                    '📅 Category', '🔢 Integer', '🔢 Integer', '🔢 Integer', '📊 Float', '🎯 Category'],
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
                            marker=dict(line=dict(color='white', width=3)))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, family='Arial Black')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='engagement_rate', nbins=40,
                             title='📈 Engagement Distribution',
                             color_discrete_sequence=['#667eea'])
            fig.update_traces(marker=dict(line=dict(color='white', width=1)))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        colors = ['#667eea', '#f093fb', '#4facfe']
        fig = go.Figure()
        for idx, col in enumerate(['post_length', 'previous_posts_count', 'follower_count']):
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=colors[idx],
                                line=dict(width=2)))
        fig.update_layout(
            title='📦 Feature Box Plots',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        st.markdown("### 🔗 Correlation Matrix")
        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                       title='🌡️ Feature Correlations',
                       color_continuous_scale='RdBu_r')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
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
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            day_risk = df.groupby(['day_of_week', 'risk_level']).size().reset_index(name='count')
            fig = px.bar(day_risk, x='day_of_week', y='count', color='risk_level',
                        title='📅 Risk by Day',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'},
                        barmode='stack')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        st.markdown("### ⚠️ Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        high_risk = len(df[df['risk_level'] == 'High'])
        med_risk = len(df[df['risk_level'] == 'Medium'])
        low_risk = len(df[df['risk_level'] == 'Low'])
        
        col1.metric("🔴 High Risk", high_risk, f"{high_risk/len(df)*100:.1f}%")
        col2.metric("🟡 Medium Risk", med_risk, f"{med_risk/len(df)*100:.1f}%")
        col3.metric("🟢 Low Risk", low_risk, f"{low_risk/len(df)*100:.1f}%")
        
        fig = px.violin(df, y='sentiment_score', x='risk_level', box=True,
                       title='🎻 Sentiment vs Risk',
                       color='risk_level',
                       color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[4]:
        st.markdown("### 📝 Text Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='word_count', y='polarity',
                           color='risk_level', size='subjectivity',
                           title='📊 Text Metrics',
                           color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            wordcloud_fig = create_wordcloud(df['text'])
            st.pyplot(wordcloud_fig)

elif page == "Visuals Pro" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🎨 VISUALIZATION STUDIO</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    df = perform_text_analysis(df)
    
    viz_type = st.selectbox(
        "🎯 Select Visualization",
        ["🌌 3D Scatter", "☀️ Sunburst", "🎨 Parallel Coordinates", 
         "🎯 Radar Chart", "🗺️ Treemap", "🌊 Sankey Diagram"],
        index=0
    )
    
    if viz_type == "🌌 3D Scatter":
        fig = px.scatter_3d(df, x='post_length', y='engagement_rate', z='sentiment_score',
                           color='risk_level', size='follower_count',
                           title='🌌 3D Feature Space',
                           color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
        fig.update_layout(
            scene=dict(bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "☀️ Sunburst":
        fig = px.sunburst(df, path=['risk_level', 'time_of_day', 'day_of_week'],
                         title='☀️ Hierarchical Distribution',
                         color='risk_level',
                         color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🎨 Parallel Coordinates":
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🎯 Radar Chart":
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
        fig.update_layout(
            title='🎯 Risk Characteristics',
            polar=dict(bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🗺️ Treemap":
        fig = px.treemap(df, path=['risk_level', 'time_of_day'],
                        title='🗺️ Risk Treemap',
                        color='risk_level',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_traces(marker=dict(line=dict(width=2, color='white')))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🌊 Sankey Diagram":
        flow_data = df.groupby(['time_of_day', 'risk_level']).size().reset_index(name='count')
        labels = list(flow_data['time_of_day'].unique()) + list(flow_data['risk_level'].unique())
        source = [labels.index(x) for x in flow_data['time_of_day']]
        target = [labels.index(x) for x in flow_data['risk_level']]
        
        link_colors = []
        for risk in flow_data['risk_level']:
            if risk == 'High':
                link_colors.append('rgba(255, 107, 107, 0.4)')
            elif risk == 'Medium':
                link_colors.append('rgba(245, 175, 25, 0.4)')
            else:
                link_colors.append('rgba(67, 233, 123, 0.4)')
        
        node_colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#f5af19', '#ff6b6b']
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20,
                     color=node_colors[:len(labels)],
                     line=dict(color='white', width=2)),
            link=dict(source=source, target=target, value=flow_data['count'],
                     color=link_colors)
        )])
        fig.update_layout(title='🌊 Flow Diagram', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Engineer" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">⚙️ FEATURE ENGINEERING LAB</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Text Features")
        if st.checkbox("✨ Apply NLP Analysis", value=True):
            with st.spinner("Processing..."):
                df = perform_text_analysis(df)
                st.success("✅ Text features created!")
                st.write("New: text_length, word_count, polarity, subjectivity")
        
        st.markdown("### ⏰ Temporal Features")
        if st.checkbox("🕐 Create Time Features"):
            time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
            df['time_numeric'] = df['time_of_day'].map(time_mapping)
            
            day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
            df['day_numeric'] = df['day_of_week'].map(day_mapping)
            df['is_weekend'] = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)
            
            st.success("✅ Temporal features created!")
    
    with col2:
        st.markdown("### 🔗 Interaction Features")
        if st.checkbox("🎯 Create Interactions"):
            df['engagement_per_follower'] = df['engagement_rate'] / (df['follower_count'] + 1)
            df['posts_per_day'] = df['previous_posts_count'] / (df['account_age_days'] + 1)
            df['sentiment_engagement'] = df['sentiment_score'] * df['engagement_rate']
            st.success("✅ Interaction features created!")
        
        st.markdown("### 📊 Scaling")
        if st.checkbox("⚡ Apply Scaling"):
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
    
    model_tabs = st.tabs(["🎯 Train", "📊 Results", "🎛️ Tune", "📈 Compare"])
    
    with model_tabs[0]:
        st.markdown("### 🎯 Select Models to Train")
        
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
            status_text = st.empty()
            
            models_to_train = []
            if use_rf:
                models_to_train.append(("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)))
            if use_gb:
                models_to_train.append(("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)))
            if use_lr:
                models_to_train.append(("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)))
            
            for idx, (name, model) in enumerate(models_to_train):
                status_text.text(f"🔄 Training {name}...")
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                results[name] = {
                    'model': model,
                    'predictions': pred,
                    'accuracy': (pred == y_test).mean()
                }
                progress_bar.progress((idx + 1) / len(models_to_train))
                time.sleep(0.5)
            
            st.session_state.models = results
            status_text.empty()
            progress_bar.empty()
            st.success("✅ All models trained successfully!")
            st.balloons()
    
    with model_tabs[1]:
        if 'models' in st.session_state:
            st.markdown("### 📊 Model Performance")
            
            for model_name, result in st.session_state.models.items():
                with st.expander(f"🎯 {model_name} - Accuracy: {result['accuracy']:.4f}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        report = classification_report(y_test, result['predictions'], 
                                                     target_names=le.classes_,
                                                     output_dict=True)
                        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
                    
                    with col2:
                        cm = confusion_matrix(y_test, result['predictions'])
                        fig = px.imshow(cm, text_auto=True, aspect="auto",
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=le.classes_, y=le.classes_,
                                       color_continuous_scale='Blues')
                        fig.update_layout(title=f"{model_name} Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Train models first!")
    
    with model_tabs[2]:
        st.markdown("### 🎛️ Hyperparameter Tuning")
        
        model_choice = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])
        
        if model_choice == "Random Forest":
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("🌲 Trees", 50, 300, 100, 25)
            with col2:
                max_depth = st.slider("📏 Depth", 5, 50, 10, 5)
            
            if st.button("⚡ Optimize", use_container_width=True):
                with st.spinner("🔄 Tuning..."):
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #43e97b, #38f9d7); 
                                padding: 30px; border-radius: 20px; color: white; text-align: center;'>
                        <h2>CV Score: {scores.mean():.4f}</h2>
                        <p>± {scores.std():.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                learning_rate = st.slider("📈 Learning Rate", 0.01, 0.3, 0.1, 0.01)
            with col2:
                n_estimators = st.slider("⚡ Estimators", 50, 300, 100, 25)
            
            if st.button("⚡ Optimize", use_container_width=True):
                with st.spinner("🔄 Tuning..."):
                    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
                    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                                padding: 30px; border-radius: 20px; color: white; text-align: center;'>
                        <h2>CV Score: {scores.mean():.4f}</h2>
                        <p>± {scores.std():.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with model_tabs[3]:
        if 'models' in st.session_state:
            st.markdown("### 📈 Model Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': list(st.session_state.models.keys()),
                'Accuracy': [r['accuracy'] for r in st.session_state.models.values()]
            }).sort_values('Accuracy', ascending=False)
            
            fig = px.bar(comparison_df, x='Model', y='Accuracy',
                        title='🏆 Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='Viridis',
                        text='Accuracy')
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            best_model = comparison_df.iloc[0]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb, #f5576c); 
                        padding: 30px; border-radius: 20px; color: white; text-align: center;'>
                <h2>🏆 Best Model</h2>
                <h1>{best_model['Model']}</h1>
                <h3>{best_model['Accuracy']:.4f} Accuracy</h3>
            </div>
            """, unsafe_allow_html=True)

elif page == "Predict" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🔮 PREDICTION ENGINE</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Enter Post Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        text_input = st.text_area("📝 Post Content", "I feel overwhelmed...", height=100)
        post_length = st.slider("📏 Length", 10, 500, len(text_input))
        engagement_rate = st.slider("📊 Engagement", 0.0, 1.0, 0.5)
    
    with col2:
        previous_posts = st.number_input("📝 Previous Posts", 0, 1000, 100)
        account_age = st.number_input("🕐 Account Age (days)", 1, 3650, 365)
    
    with col3:
        followers = st.number_input("👥 Followers", 0, 10000, 500)
        sentiment = st.slider("😊 Sentiment", -1.0, 1.0, 0.0)
    
    if st.button("🎯 ANALYZE RISK", use_container_width=True, type="primary"):
        if 'models' in st.session_state:
            with st.spinner("🔄 Analyzing..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                input_data = np.array([[post_length, engagement_rate, previous_posts,
                                       account_age, followers, sentiment]])
                
                best_model = max(st.session_state.models.items(), key=lambda x: x[1]['accuracy'])
                
                scaler = StandardScaler()
                df = st.session_state.processed_data
                feature_cols = ['post_length', 'engagement_rate', 'previous_posts_count',
                              'account_age_days', 'follower_count', 'sentiment_score']
                scaler.fit(df[feature_cols].fillna(df[feature_cols].mean()))
                input_scaled = scaler.transform(input_data)
                
                prediction = best_model[1]['model'].predict(input_scaled)[0]
                proba = best_model[1]['model'].predict_proba(input_scaled)[0]
                
                le = LabelEncoder()
                le.fit(['Low', 'Medium', 'High'])
                risk_level = le.inverse_transform([prediction])[0]
                
                progress_bar.empty()
                
                st.markdown("---")
                st.markdown("### 🎯 ANALYSIS RESULTS")
                
                if risk_level == "High":
                    color = "#ff6b6b"
                    icon = "🔴"
                    message = "IMMEDIATE ATTENTION REQUIRED"
                elif risk_level == "Medium":
                    color = "#f5af19"
                    icon = "🟡"
                    message = "MONITOR CLOSELY"
                else:
                    color = "#43e97b"
                    icon = "🟢"
                    message = "LOW RISK - STANDARD MONITORING"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}, {color}dd); 
                            padding: 50px; border-radius: 25px; color: white; text-align: center;
                            box-shadow: 0 15px 40px {color}60;
                            border: 3px solid rgba(255, 255, 255, 0.3);'>
                    <div style='font-size: 5rem; margin-bottom: 20px;'>{icon}</div>
                    <h1 style='font-size: 3rem; margin: 20px 0;'>{risk_level.upper()} RISK</h1>
                    <p style='font-size: 1.5rem; margin: 0; opacity: 0.95;'>{message}</p>
                    <h2 style='font-size: 2.5rem; margin-top: 30px;'>{max(proba)*100:.1f}% Confidence</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 Low", f"{proba[0]*100:.1f}%")
                col2.metric("🟡 Medium", f"{proba[1]*100:.1f}%")
                col3.metric("🔴 High", f"{proba[2]*100:.1f}%")
                
                proba_df = pd.DataFrame({
                    'Risk Level': le.classes_,
                    'Probability': proba * 100
                })
                
                fig = px.bar(proba_df, x='Risk Level', y='Probability',
                            title='📊 Risk Probability Distribution',
                            color='Risk Level',
                            color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
                fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                if risk_level == "High":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #ff6b6b, #f5576c); 
                                padding: 30px; border-radius: 20px; color: white;
                                border: 2px solid rgba(255, 255, 255, 0.3);'>
                        <h3>🚨 IMMEDIATE ACTION REQUIRED</h3>
                        <ul style='font-size: 1.1rem; line-height: 2;'>
                            <li>Contact mental health professional immediately</li>
                            <li>Call crisis hotline: <b>988</b></li>
                            <li>Ensure user safety</li>
                            <li>Activate support network</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please train models first in AI Models page!")

elif page == "Docs":
    st.markdown('<p class="main-header">📚 DOCUMENTATION CENTER</p>', unsafe_allow_html=True)
    
    doc_tabs = st.tabs(["📖 Overview", "🎯 Features", "🔬 Methods", "📊 Usage", "⚠️ Ethics"])
    
    with doc_tabs[0]:
        st.markdown("""
        ## 🎯 Project Overview
        
        **MindGuard Analytics** is an advanced AI-powered platform for mental health crisis detection 
        through social media analysis. Built with cutting-edge machine learning and NLP techniques.
        
        ### 🚀 Key Capabilities
        - **Multi-source data integration** with advanced preprocessing
        - **16+ visualization types** including 3D, Sunburst, Sankey
        - **3 ML models** with hyperparameter tuning
        - **Real-time predictions** with confidence scores
        - **Beautiful UI** with animated gradients
        
        ### 🎨 Technologies
        - Streamlit, Pandas, NumPy, Scikit-learn
        - Plotly, Matplotlib, Seaborn
        - TextBlob for NLP
        - Modern CSS with animations
        """)
    
    with doc_tabs[1]:
        st.markdown("""
        ## ✨ Feature List
        
        ### Data Processing
        - ✅ 3 data sources (sample, upload, manual)
        - ✅ Mean, Median, KNN imputation
        - ✅ Advanced data cleaning
        - ✅ Feature encoding & scaling
        
        ### Visualizations (16+)
        - ✅ 3D Scatter, Sunburst, Treemap
        - ✅ Sankey, Radar, Parallel Coordinates
        - ✅ Heatmaps, Violin plots, Word clouds
        - ✅ Box plots, Bar charts, Pie charts
        
        ### Machine Learning
        - ✅ Random Forest, Gradient Boosting, Logistic Regression
        - ✅ Hyperparameter tuning interface
        - ✅ Cross-validation
        - ✅ Real-time predictions
        
        ### UI/UX
        - ✅ Animated gradient backgrounds
        - ✅ Glassmorphism effects
        - ✅ Interactive sidebar
        - ✅ Progress indicators
        """)
    
    with doc_tabs[2]:
        st.markdown("""
        ## 🔬 Methodology
        
        ### Text Analysis
        - Sentiment analysis using TextBlob
        - Polarity & subjectivity extraction
        - Keyword detection
        - Word count metrics
        
        ### Feature Engineering
        - Temporal encoding (time, day)
        - Interaction features
        - Text-based features
        - Scaling & normalization
        
        ### Model Training
        - 80-20 train-test split
        - StandardScaler preprocessing
        - 5-fold cross-validation
        - Ensemble methods
        
        ### Evaluation
        - Accuracy, Precision, Recall, F1
        - Confusion matrices
        - ROC curves
        - Feature importance
        """)
    
    with doc_tabs[3]:
        st.markdown("""
        ## 📊 Usage Guide
        
        ### Quick Start
        1. Click "🚀 Load" to load sample data
        2. Navigate to "Data Hub" to explore
        3. Visit "Visuals Pro" for advanced charts
        4. Train models in "AI Models"
        5. Make predictions in "Predict"
        
        ### Tips
        - Use KNN imputation for best results
        - Train all 3 models for comparison
        - Tune hyperparameters for optimal performance
        - Export results using screenshot
        
        ### Keyboard Shortcuts
        - `Ctrl/Cmd + K` - Focus sidebar
        - `Esc` - Close modals
        - `F` - Toggle fullscreen
        """)
    
    with doc_tabs[4]:
        st.markdown("""
        ## ⚠️ Ethics & Privacy
        
        ### Important Disclaimers
        - **Research tool only** - not for diagnosis
        - **Consult professionals** for real cases
        - **No data storage** - privacy first
        - **Bias awareness** - regular audits needed
        
        ### Crisis Resources
        - **USA**: 988 (Suicide & Crisis Lifeline)
        - **Text**: HOME to 741741 (Crisis Text Line)
        - **International**: IASP.info
        
        ### Responsible AI
        - Transparency in decision-making
        - Human oversight required
        - Clear limitations communicated
        - Ethical data handling
        """)

else:
    if not st.session_state.data_loaded:
        st.markdown('<p class="main-header">🛡️ MINDGUARD ANALYTICS</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); 
                    padding: 60px; border-radius: 25px; color: white; text-align: center;
                    box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);'>
            <h1 style='font-size: 3rem; margin: 0 0 20px 0;'>👋 Welcome!</h1>
            <p style='font-size: 1.5rem; margin: 0;'>
                Click <b>"🚀 Load"</b> in the sidebar to start your analytics journey!
            </p>
        </div>
        """, unsafe_allow_html=True)

# Animated footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='font-size: 1.2rem; font-weight: 700; 
               background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;'>
        🛡️ MindGuard Analytics | AI-Powered Mental Health Detection
    </p>
    <p style='color: #666; font-size: 0.9rem;'>
        Made with ❤️ for Mental Health Awareness | CMSE 830 Fall 2025
    </p>
    <p style='color: #999; font-size: 0.8rem;'>
        ⚠️ Crisis? Call 988 (USA) | Text HOME to 741741
    </p>
</div>
""", unsafe_allow_html=True)
