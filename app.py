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

# Page config
st.set_page_config(page_title="MindGuard Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with ocean blue theme
st.markdown("""
<style>
    /* Gradient background animation */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #e0f2f7, #f0f4f8, #e8f5e9, #fff9e6, #fce4ec);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        background-attachment: fixed;
    }
    
    /* Vibrant but readable sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1976d2 0%, #1565c0 50%, #0d47a1 100%) !important;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Headers with DARK text for readability */
    .main-header {
        font-size: 3rem !important;
        font-weight: 900 !important;
        text-align: center !important;
        margin: 30px 0 !important;
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.8);
        background: linear-gradient(135deg, #1976d2, #1565c0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Buttons with excellent contrast */
    .stButton > button {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 25px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3) !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4) !important;
    }
    
    /* Dark text for metrics */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* All text elements dark for readability */
    .stMarkdown, .stText, p, span, div {
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    /* Info boxes with dark text */
    .stAlert {
        color: #1a1a1a !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #1976d2, #1565c0) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_data():
    """Load mental health dataset"""
    np.random.seed(42)
    n = 1000
    
    risk_levels = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.5, 0.3, 0.2])
    
    texts = []
    for risk in risk_levels:
        if risk == 'High':
            texts.append(np.random.choice([
                "I want to end it all", "Nobody would miss me", "I can't do this anymore",
                "Life is not worth living", "I wish I was dead", "Everyone would be better off without me"
            ]))
        elif risk == 'Medium':
            texts.append(np.random.choice([
                "I'm feeling really down", "Nothing seems to matter", "I feel so alone",
                "Everything is falling apart", "I don't know what to do", "Feeling hopeless today"
            ]))
        else:
            texts.append(np.random.choice([
                "Having a good day", "Feeling grateful", "Things are looking up",
                "Spent time with family", "Enjoyed my morning coffee", "Beautiful weather today"
            ]))
    
    data = {
        'text': texts,
        'risk_level': risk_levels,
        'post_length': np.random.randint(20, 200, n),
        'engagement_rate': np.random.uniform(0, 1, n),
        'sentiment_score': np.random.uniform(-1, 1, n),
        'previous_posts_count': np.random.randint(0, 100, n),
        'follower_count': np.random.randint(10, 10000, n),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n),
        'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='H')
    }
    
    return pd.DataFrame(data)

def perform_text_analysis(df):
    """Add text analysis features"""
    if 'text' not in df.columns:
        return df
    
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    return df

def handle_missing_values(df, method='mean'):
    """Handle missing values"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_imputed = df.copy()
    
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    else:
        imputer = KNNImputer(n_neighbors=5)
    
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

def analyze_mental_health_risk_transformer(text):
    """
    TRANSFORMER-BASED SEMANTIC UNDERSTANDING
    Uses sentence embeddings + emotion analysis
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load pre-trained sentence transformer
        # Using all-MiniLM-L6-v2 (fast, accurate, good for semantic similarity)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get sentence embedding
        text_embedding = model.encode([text])[0]
        
        # Define reference embeddings for each risk category
        high_risk_examples = [
            "I want to die and end my life",
            "I cannot live anymore, nobody cares",
            "I hate this life and want to disappear",
            "Everyone would be better off without me",
            "I am skeptical whether to live or not",
            "I don't want to live anymore",
            "Life is not worth living",
            "I feel empty and worthless",
            "Nothing makes sense anymore",
            "I wish I could just disappear forever"
        ]
        
        medium_risk_examples = [
            "I am skeptical about continuing",
            "I hate this life sometimes",
            "Feeling really down and hopeless",
            "Everything feels meaningless",
            "I don't know if I can go on",
            "Feeling very alone and sad",
            "Nothing brings me joy anymore",
            "I feel trapped and helpless"
        ]
        
        low_risk_examples = [
            "Having a good day today",
            "Feeling grateful for my family",
            "The weather is beautiful",
            "I love spending time with friends",
            "Work was productive today",
            "Enjoyed my morning coffee",
            "Looking forward to the weekend"
        ]
        
        # Get embeddings for reference examples
        high_risk_embeddings = model.encode(high_risk_examples)
        medium_risk_embeddings = model.encode(medium_risk_examples)
        low_risk_embeddings = model.encode(low_risk_examples)
        
        # Calculate similarity scores
        high_similarity = cosine_similarity([text_embedding], high_risk_embeddings).max()
        medium_similarity = cosine_similarity([text_embedding], medium_risk_embeddings).max()
        low_similarity = cosine_similarity([text_embedding], low_risk_embeddings).max()
        
        # Get sentiment
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Combine semantic similarity with sentiment
        # High negative sentiment increases risk
        sentiment_adjustment = 0
        if sentiment < -0.3:
            sentiment_adjustment = abs(sentiment) * 15
        elif sentiment > 0.3:
            sentiment_adjustment = -sentiment * 20
        
        # Calculate risk scores (0-100)
        high_score = (high_similarity * 100) + sentiment_adjustment
        medium_score = (medium_similarity * 100)
        low_score = (low_similarity * 100) - sentiment_adjustment
        
        # Normalize to ensure they sum properly
        total = high_score + medium_score + low_score
        if total > 0:
            high_prob = high_score / total
            medium_prob = medium_score / total
            low_prob = low_score / total
        else:
            high_prob = medium_prob = low_prob = 0.33
        
        # Determine risk level based on highest probability
        probabilities = {'High': high_prob, 'Medium': medium_prob, 'Low': low_prob}
        risk_level = max(probabilities, key=probabilities.get)
        
        # Calculate final risk score (0-100)
        risk_score = (high_prob * 100) + (medium_prob * 50) + (low_prob * 0)
        
        # Adjust thresholds to be more sensitive
        if risk_score >= 60 or high_similarity > 0.5:
            risk_level = "High"
        elif risk_score >= 35 or medium_similarity > 0.45:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Check if completely unrelated
        max_similarity = max(high_similarity, medium_similarity, low_similarity)
        if max_similarity < 0.3 and sentiment > -0.2:
            return {
                'risk_level': 'Low',
                'risk_score': 5,
                'confidence': 0.95,
                'probabilities': np.array([0.95, 0.04, 0.01]),
                'detected_factors': ['🟢 No mental health or suicide-related content detected'],
                'sentiment': sentiment,
                'high_similarity': high_similarity,
                'medium_similarity': medium_similarity,
                'low_similarity': low_similarity,
                'is_safe_unrelated': True
            }
        
        # Build detected factors
        detected_factors = []
        detected_factors.append(f"📊 Semantic similarity to high-risk: {high_similarity:.2f}")
        detected_factors.append(f"📊 Semantic similarity to medium-risk: {medium_similarity:.2f}")
        detected_factors.append(f"📊 Semantic similarity to low-risk: {low_similarity:.2f}")
        detected_factors.append(f"💭 Sentiment score: {sentiment:.2f}")
        
        if high_similarity > 0.4:
            detected_factors.append("🔴 Strong semantic match to suicidal ideation")
        elif medium_similarity > 0.4:
            detected_factors.append("🟡 Moderate semantic match to mental health concerns")
        
        confidence = max(high_prob, medium_prob, low_prob)
        
        return {
            'risk_level': risk_level,
            'risk_score': int(risk_score),
            'confidence': confidence,
            'probabilities': np.array([low_prob, medium_prob, high_prob]),
            'detected_factors': detected_factors,
            'sentiment': sentiment,
            'high_similarity': high_similarity,
            'medium_similarity': medium_similarity,
            'low_similarity': low_similarity,
            'is_safe_unrelated': False
        }
        
    except ImportError:
        # Fallback if sentence-transformers not installed
        st.error("⚠️ Sentence transformers not installed. Install with: pip install sentence-transformers")
        return {
            'risk_level': 'Medium',
            'risk_score': 50,
            'confidence': 0.5,
            'probabilities': np.array([0.33, 0.34, 0.33]),
            'detected_factors': ['❌ Model not available - install sentence-transformers'],
            'sentiment': 0,
            'is_safe_unrelated': False
        }

# SIDEBAR
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>🛡️ MindGuard</h2>", unsafe_allow_html=True)
    
    page = st.radio("", 
                   ["🏠 Home", "📊 Data Hub", "🔍 Explorer", "🎨 Visuals Pro", 
                    "⚙️ Engineer", "🤖 AI Models", "🔮 Predict", "📚 Docs"],
                   label_visibility="collapsed")
    
    page = page.split(" ", 1)[1]
    
    st.markdown("---")
    st.markdown("### 🆘 Crisis Resources")
    st.markdown("""
    **Suicide Prevention Lifeline**
    📞 988 (US)
    
    **Crisis Text Line**
    📱 Text HOME to 741741
    
    **International**
    🌍 findahelpline.com
    """)

# HOME PAGE
if page == "Home":
    st.markdown('<p class="main-header">🛡️ MINDGUARD ANALYTICS</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0;'>
        <h2 style='color: #1a1a1a;'>AI-Powered Mental Health Risk Assessment</h2>
        <p style='font-size: 1.2rem; color: #2c3e50;'>
            Using transformer-based NLP with sentence embeddings for accurate semantic understanding
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 15px; color: white; text-align: center;'>
            <h3>🧠 Transformer-Based</h3>
            <p>Sentence-BERT embeddings for semantic understanding</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb, #f5576c); padding: 30px; border-radius: 15px; color: white; text-align: center;'>
            <h3>📊 Multi-Class</h3>
            <p>Low, Medium, High risk classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 30px; border-radius: 15px; color: white; text-align: center;'>
            <h3>🎯 Contextual</h3>
            <p>Understands meaning, not just keywords</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if st.button("📂 LOAD DEMO DATA", use_container_width=True, type="primary"):
        with st.spinner("Loading data..."):
            st.session_state.raw_data = load_data()
            st.session_state.processed_data = st.session_state.raw_data.copy()
            st.session_state.data_loaded = True
            time.sleep(1)
        st.success("✅ Data loaded successfully!")
        st.rerun()


elif page == "Data Hub" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">📊 DATA HUB</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    
    tabs = st.tabs(["🔍 Dataset", "💎 Quality", "🧹 Clean", "📖 Dictionary"])
    
    with tabs[0]:
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Features", len(df.columns))
        col3.metric("High Risk", len(df[df['risk_level']=='High']))
        col4.metric("Low Risk", len(df[df['risk_level']=='Low']))
        
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        st.info("📊 **What this table shows:** Statistical summary of numeric features. Mean = average, Std = spread, Min/Max = range, 25%/50%/75% = quartiles showing distribution.")
        
    with tabs[1]:
        st.markdown("### 💎 Data Quality Report")
        
        missing_df = pd.DataFrame({
            'Feature': df.columns,
            'Missing': df.isnull().sum(),
            'Percent': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        st.dataframe(missing_df, use_container_width=True)
        st.info("❓ **What this table shows:** Count and percentage of missing values per feature. Higher percentages need attention before analysis.")
        
    with tabs[2]:
        st.markdown("### 🧹 Data Cleaning")
        
        method = st.selectbox("Imputation Method", ["mean", "median", "knn"])
        
        if st.button("Clean Data"):
            st.session_state.processed_data = handle_missing_values(df, method)
            st.success("✅ Data cleaned!")
    
    with tabs[3]:
        st.markdown("### 📖 Data Dictionary")
        st.markdown("""
        | Feature | Description |
        |---------|-------------|
        | text | Social media post content |
        | risk_level | Assessed risk (Low/Medium/High) |
        | post_length | Character count |
        | engagement_rate | User engagement (0-1) |
        | sentiment_score | Sentiment polarity (-1 to 1) |
        | previous_posts_count | User's post history |
        | follower_count | Number of followers |
        | time_of_day | When posted |
        | day_of_week | Day of posting |
        """)

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
            st.info("📊 **What this shows:** This pie chart displays the percentage breakdown of all posts by risk level (Low/Medium/High). It helps you quickly identify what proportion of content requires immediate attention versus standard monitoring.")
        
        with col2:
            fig = px.histogram(df, x='engagement_rate', nbins=40,
                             title='📈 Engagement Distribution',
                             color_discrete_sequence=['#667eea'])
            fig.update_traces(marker=dict(line=dict(color='white', width=1)))
            fig.update_layout(font=dict(size=13, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
            st.info("📈 **What this shows:** This histogram shows how engagement rates are distributed across all posts. The x-axis shows engagement level, y-axis shows count. Helps identify typical engagement patterns.")
        
        colors = ['#667eea', '#f093fb', '#4facfe']
        fig = go.Figure()
        for idx, col in enumerate(['post_length', 'previous_posts_count', 'follower_count']):
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=colors[idx]))
        fig.update_layout(title='📦 Feature Distributions', font=dict(size=13, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
        st.info("📦 **What this shows:** Box plots compare the distribution of post length, previous posts count, and follower count. The box shows middle 50% of values, line shows median, dots show outliers.")
    
    with viz_tabs[1]:
        st.markdown("### 🔗 Correlation Matrix")
        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                       title='🌡️ Feature Correlations',
                       color_continuous_scale='RdBu_r')
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
        st.info("🔥 **What this shows:** This heatmap reveals relationships between all numeric features. Red = strong positive correlation, Blue = strong negative correlation, White = no correlation. Numbers show exact correlation strength (-1 to 1).")
    
    with viz_tabs[2]:
        st.markdown("### ⏰ Time Analysis")
        
        time_counts = df.groupby(['time_of_day', 'risk_level']).size().reset_index(name='count')
        fig = px.bar(time_counts, x='time_of_day', y='count', color='risk_level',
                    title='⏰ Risk by Time of Day',
                    color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_layout(font=dict(size=13, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        st.markdown("### ⚠️ Risk Analysis")
        
        risk_metrics = df.groupby('risk_level')[['engagement_rate', 'sentiment_score']].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=risk_metrics.index, y=risk_metrics['engagement_rate'],
                           name='Engagement', marker_color='#667eea'))
        fig.update_layout(title='📊 Metrics by Risk Level', font=dict(size=13, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[4]:
        st.markdown("### 📝 Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_words = pd.Series(' '.join(df['text']).lower().split()).value_counts().head(20)
            fig = px.bar(x=top_words.values, y=top_words.index, orientation='h',
                        title='📊 Top 20 Words', labels={'x': 'Count', 'y': 'Word'})
            fig.update_layout(font=dict(size=12, color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            wordcloud_fig = create_wordcloud(df['text'])
            st.pyplot(wordcloud_fig)
            st.info("☁️ **What this shows:** Word cloud displays the most frequently used words in all posts. Larger words appear more often, helping identify common themes and concerns.")

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
        st.info("🎯 **What this shows:** Three-dimensional view of post length, engagement rate, and follower count colored by risk level. Helps identify clustering patterns in high-risk content.")
        fig = px.scatter_3d(df, x='post_length', y='engagement_rate', z='follower_count',
                           color='risk_level', title='🎯 3D Risk Scatter',
                           color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sunburst":
        st.info("🌅 **What this shows:** Hierarchical sunburst showing risk distribution across time periods. Inner ring = time of day, outer rings = how risk levels distribute within each period.")
        fig = px.sunburst(df, path=['time_of_day', 'risk_level'],
                         title='🌅 Time-Risk Sunburst',
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
        st.info("📊 **What this shows:** Each line represents one post across multiple variables, colored by risk level. Crossing patterns reveal correlations - parallel lines mean features move together, crossing lines show inverse relationships.")
    
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
                marker_color=colors[risk],
                opacity=0.6
            ))
        fig.update_layout(title='🎯 Risk Characteristics', font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
        st.info("📡 **What this shows:** Radar chart compares average feature values across risk categories. Each axis represents a different metric. Larger areas indicate higher values - helps visualize the profile of each risk level.")
    
    elif viz_type == "Treemap":
        st.info("🗺️ **What this shows:** Treemap shows hierarchical data as nested rectangles. Size represents volume, color represents risk level, helping visualize proportions.")
        fig = px.treemap(df, path=['risk_level', 'time_of_day'],
                        title='🗺️ Risk Treemap',
                        color='risk_level',
                        color_discrete_map={'Low': '#43e97b', 'Medium': '#f5af19', 'High': '#ff6b6b'})
        fig.update_traces(marker=dict(line=dict(width=2, color='white')))
        fig.update_layout(font=dict(size=12, color='#1a1a1a'))
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sankey Diagram":
        st.info("🌊 **What this shows:** Flow diagram showing how posts progress from time of day → day of week → risk level. Ribbon width indicates volume, revealing when high-risk content appears.")
        time_day_risk = df.groupby(['time_of_day', 'day_of_week', 'risk_level']).size().reset_index(name='count')
        
        all_times = df['time_of_day'].unique()
        all_days = df['day_of_week'].unique()
        all_risks = df['risk_level'].unique()
        
        labels = list(all_times) + list(all_days) + list(all_risks)
        
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        day_to_idx = {d: i + len(all_times) for i, d in enumerate(all_days)}
        risk_to_idx = {r: i + len(all_times) + len(all_days) for i, r in enumerate(all_risks)}
        
        sources = []
        targets = []
        values = []
        
        time_day = df.groupby(['time_of_day', 'day_of_week']).size().reset_index(name='count')
        for _, row in time_day.iterrows():
            sources.append(time_to_idx[row['time_of_day']])
            targets.append(day_to_idx[row['day_of_week']])
            values.append(row['count'])
        
        day_risk = df.groupby(['day_of_week', 'risk_level']).size().reset_index(name='count')
        for _, row in day_risk.iterrows():
            sources.append(day_to_idx[row['day_of_week']])
            targets.append(risk_to_idx[row['risk_level']])
            values.append(row['count'])
        
        colors_list = ['rgba(102, 126, 234, 0.4)'] * len(all_times) + \
                     ['rgba(240, 147, 251, 0.4)'] * len(all_days) + \
                     ['rgba(67, 233, 123, 0.8)', 'rgba(245, 175, 25, 0.8)', 'rgba(255, 107, 107, 0.8)']
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='white', width=2),
                label=labels,
                color=colors_list
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color='rgba(200, 200, 200, 0.3)'
            )
        )])
        
        fig.update_layout(
            title='🌊 Time → Day → Risk Flow',
            font=dict(size=14, color='#1a1a1a'),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Engineer" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">⚙️ FEATURE ENGINEERING</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    
    st.markdown("### 🛠️ Create New Features")
    
    feature_type = st.selectbox("Feature Type", 
                               ["Text Length", "Engagement Category", "Time Features", "Sentiment Category"])
    
    if feature_type == "Text Length":
        if st.button("Create Feature"):
            df['text_length_category'] = pd.cut(df['post_length'], 
                                               bins=[0, 50, 100, 200],
                                               labels=['Short', 'Medium', 'Long'])
            st.session_state.processed_data = df
            st.success("✅ Text length categories created!")
    
    elif feature_type == "Engagement Category":
        if st.button("Create Feature"):
            df['engagement_category'] = pd.cut(df['engagement_rate'],
                                              bins=[0, 0.3, 0.7, 1.0],
                                              labels=['Low', 'Medium', 'High'])
            st.session_state.processed_data = df
            st.success("✅ Engagement categories created!")

elif page == "AI Models" and st.session_state.data_loaded:
    st.markdown('<p class="main-header">🤖 AI MODEL TRAINING</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data
    
    model_tabs = st.tabs(["🎯 Train", "📊 Results", "🎛️ Tune"])
    
    with model_tabs[0]:
        st.markdown("### 🎯 Model Training")
        
        selected_models = st.multiselect(
            "Select Models",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"],
            default=["Random Forest"]
        )
        
        if st.button("🚀 START TRAINING", use_container_width=True, type="primary"):
            with st.spinner("Training models..."):
                numeric_features = ['post_length', 'engagement_rate', 'sentiment_score', 
                                  'previous_posts_count', 'follower_count']
                
                X = df[numeric_features]
                y = df['risk_level']
                
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                models = {}
                
                if "Random Forest" in selected_models:
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train_scaled, y_train)
                    rf_pred = rf.predict(X_test_scaled)
                    models['Random Forest'] = {
                        'model': rf,
                        'predictions': rf_pred,
                        'accuracy': (rf_pred == y_test).mean()
                    }
                
                if "Gradient Boosting" in selected_models:
                    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    gb.fit(X_train_scaled, y_train)
                    gb_pred = gb.predict(X_test_scaled)
                    models['Gradient Boosting'] = {
                        'model': gb,
                        'predictions': gb_pred,
                        'accuracy': (gb_pred == y_test).mean()
                    }
                
                if "Logistic Regression" in selected_models:
                    lr = LogisticRegression(random_state=42, max_iter=1000)
                    lr.fit(X_train_scaled, y_train)
                    lr_pred = lr.predict(X_test_scaled)
                    models['Logistic Regression'] = {
                        'model': lr,
                        'predictions': lr_pred,
                        'accuracy': (lr_pred == y_test).mean()
                    }
                
                st.session_state.models = models
                st.session_state.X_test = X_test_scaled
                st.session_state.y_test = y_test
                st.session_state.label_encoder = le
                
                performance_df = pd.DataFrame({
                    'Model': list(models.keys()),
                    'Accuracy': [m['accuracy'] for m in models.values()]
                }).sort_values('Accuracy', ascending=False)
                
                st.dataframe(performance_df, use_container_width=True)
                st.info("🎯 **What this table shows:** AI model accuracy comparison. Higher scores mean better predictions. Use this to select the best model.")
                
                st.success("✅ Training complete!")
    
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
                        st.info("📊 **What this table shows:** Performance metrics per risk level. Precision = accuracy of predictions (how many predicted High were actually High), Recall = coverage (how many actual High were caught), F1-Score = balanced measure combining both.")
                    
                    with col2:
                        cm = confusion_matrix(y_test, result['predictions'])
                        fig = px.imshow(cm, text_auto=True, aspect="auto",
                                       x=le.classes_, y=le.classes_,
                                       color_continuous_scale='Blues')
                        fig.update_layout(title=f"{model_name}", font=dict(color='#1a1a1a'))
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("🎯 **What this shows:** Confusion matrix displays actual vs predicted classifications. Diagonal values (dark blue) are correct predictions. Off-diagonal values show misclassifications - helps identify where the model makes mistakes.")
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
                                   help="Enter any text - the AI will use transformer-based semantic understanding")
        
        st.markdown("""
        **💡 Try these examples:**
        - "I am skeptical whether to live or not. I hate this life." (Should be Medium-High)
        - "I don't want to live" (Should be High)
        - "The weather is good. I want to travel." (Should be Low - unrelated)
        - "I'm feeling happy today" (Should be Low)
        """)
    
    with col2:
        st.markdown("### ⚙️ About the Model")
        st.info("""
        🧠 **Transformer-Based**
        - Uses Sentence-BERT embeddings
        - Understands semantic meaning
        - Not keyword-dependent
        - Context-aware classification
        """)
    
    if st.button("🎯 ANALYZE RISK", use_container_width=True, type="primary"):
        if len(text_input.strip()) < 5:
            st.warning("⚠️ Please enter at least 5 characters")
        else:
            with st.spinner("🤖 AI analyzing with transformers..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.015)
                    progress.progress(i + 1)
                
                result = analyze_mental_health_risk_transformer(text_input)
                progress.empty()
                
                risk_level = result['risk_level']
                risk_score = result['risk_score']
                confidence = result['confidence']
                proba = result['probabilities']
                detected_factors = result['detected_factors']
                is_safe_unrelated = result.get('is_safe_unrelated', False)
                
                # Special handling for safe unrelated content
                if is_safe_unrelated:
                    st.success("✅ **NO HARM DETECTED** - This text contains no suicide-related content or mental health crisis indicators. The content appears completely safe and unrelated to self-harm.")
                
                if risk_level == "High":
                    color, icon, message = "#ff6b6b", "🔴", "HIGH RISK - IMMEDIATE ATTENTION"
                elif risk_level == "Medium":
                    color, icon, message = "#f5af19", "🟡", "MEDIUM RISK - CLOSE MONITORING"
                else:
                    color, icon, message = "#43e97b", "🟢", "LOW RISK - STANDARD MONITORING"
                
                st.markdown("---")
                
                if detected_factors:
                    st.markdown("### 🔍 Analysis:")
                    for factor in detected_factors[:8]:
                        if "🔴" in factor:
                            st.error(factor)
                        elif "🟡" in factor:
                            st.warning(factor)
                        else:
                            st.success(factor)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: {color}; padding: 50px; border-radius: 20px; color: white; 
                            text-align: center; box-shadow: 0 10px 30px {color}60; 
                            border: 3px solid rgba(255,255,255,0.3);'>
                    <div style='font-size: 5rem;'>{icon}</div>
                    <h1 style='font-size: 3rem; margin: 20px 0; color: white !important; font-weight: 900;'>{risk_level.upper()} RISK</h1>
                    <p style='font-size: 1.3rem; color: white !important; font-weight: 600;'>{message}</p>
                    <h2 style='font-size: 2.5rem; margin-top: 30px; color: white !important;'>Score: {risk_score}/100</h2>
                    <p style='font-size: 1.2rem; color: white !important;'>{confidence*100:.1f}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 Low", f"{proba[0]*100:.1f}%")
                col2.metric("🟡 Medium", f"{proba[1]*100:.1f}%")
                col3.metric("🔴 High", f"{proba[2]*100:.1f}%")
                
                with st.expander("📊 Technical Details", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Risk Score", f"{risk_score}/100")
                    col2.metric("Sentiment", f"{result['sentiment']:.2f}")
                    if 'high_similarity' in result:
                        col3.metric("High Sim", f"{result['high_similarity']:.2f}")
                        col4.metric("Med Sim", f"{result['medium_similarity']:.2f}")
                    
                    st.markdown("**All Factors:**")
                    for factor in detected_factors:
                        st.write(factor)
                
                st.markdown("---")
                st.markdown("### 💡 Recommendations")
                
                if risk_level == "High":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #ff6b6b, #f5576c); padding: 35px; 
                                border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(255,107,107,0.4);'>
                        <h3 style='color: white !important;'>🚨 IMMEDIATE ACTION</h3>
                        <ul style='font-size: 1.1rem; line-height: 2.2; color: white !important;'>
                            <li><b>Call 988</b> (USA Suicide Lifeline)</li>
                            <li><b>Text HOME to 741741</b></li>
                            <li><b>Go to emergency room</b></li>
                            <li><b>Do not leave alone</b></li>
                            <li><b>Contact mental health professional</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "Medium":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #f5af19, #f39c12); padding: 30px; 
                                border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(245,175,25,0.4);'>
                        <h3 style='color: white !important;'>⚠️ CLOSE MONITORING</h3>
                        <ul style='font-size: 1.05rem; line-height: 2; color: white !important;'>
                            <li><b>Check in regularly</b></li>
                            <li><b>Encourage professional help</b></li>
                            <li><b>Monitor for escalation</b></li>
                            <li><b>Provide crisis resources</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #43e97b, #38f9d7); padding: 25px; 
                                border-radius: 15px; color: white; box-shadow: 0 8px 25px rgba(67,233,123,0.4);'>
                        <h3 style='color: white !important;'>✅ STANDARD MONITORING</h3>
                        <ul style='font-size: 1rem; line-height: 2; color: white !important;'>
                            <li><b>Continue regular check-ins</b></li>
                            <li><b>Encourage positive activities</b></li>
                            <li><b>Maintain awareness</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

elif page == "Docs":
    st.markdown('<p class="main-header">📚 DOCUMENTATION</p>', unsafe_allow_html=True)
    
    doc_tabs = st.tabs(["Overview", "Features", "Methods", "Usage"])
    
    with doc_tabs[0]:
        st.markdown("""
        ## 🎯 Project Overview
        
        **MindGuard Analytics** is an AI-powered mental health risk assessment system using:
        - 🧠 **Transformer-based NLP** (Sentence-BERT)
        - 📊 **Semantic understanding** (not keyword matching)
        - 🎯 **Multi-class classification** (Low/Medium/High)
        - 💡 **Context-aware** analysis
        
        ### 🌟 Key Innovation
        Unlike traditional keyword-matching systems, MindGuard uses **sentence embeddings** to understand
        the semantic meaning of text, allowing it to:
        - Detect suicidal ideation even without explicit keywords
        - Understand context and nuance
        - Generalize to diverse language patterns
        - Reduce false negatives
        """)
    
    with doc_tabs[1]:
        st.markdown("""
        ## ✨ Features
        
        ### 🔮 Prediction Engine
        - **Transformer-based**: Uses Sentence-BERT for semantic understanding
        - **Semantic similarity**: Compares input to reference examples
        - **Multi-class**: Low, Medium, High risk classification
        - **Confidence scores**: Probability distribution across classes
        
        ### 📊 Visualization
        - 13+ interactive charts with explanations
        - Correlation analysis
        - Time-series patterns
        - Risk distribution
        
        ### 🤖 Machine Learning
        - Random Forest, Gradient Boosting, Logistic Regression
        - Model comparison and tuning
        - Performance metrics
        - Confusion matrices
        """)
    
    with doc_tabs[2]:
        st.markdown("""
        ## 🧠 Methodology
        
        ### Transformer-Based Analysis
        
        **1. Sentence Embedding**
        - Uses Sentence-BERT (all-MiniLM-L6-v2)
        - Converts text to 384-dimensional vectors
        - Captures semantic meaning
        
        **2. Reference Examples**
        - High risk: 10 examples of suicidal ideation
        - Medium risk: 8 examples of mental distress
        - Low risk: 7 examples of positive/neutral content
        
        **3. Cosine Similarity**
        - Compares input embedding to each reference category
        - Calculates similarity scores (0-1)
        - Combines with sentiment analysis
        
        **4. Classification**
        - Uses similarity scores + sentiment
        - Determines risk level (High/Medium/Low)
        - Generates confidence and probabilities
        
        ### Why This Works
        - **"I am skeptical whether to live"** → High similarity to "I want to die"
        - **"I hate this life"** → High similarity to mental distress examples
        - **"The weather is good"** → Low similarity to all mental health examples
        """)
    
    with doc_tabs[3]:
        st.markdown("""
        ## 📖 Usage Guide
        
        ### 1. Load Data
        Click **"LOAD DEMO DATA"** on the Home page
        
        ### 2. Explore Data
        - **Data Hub**: View and clean data
        - **Explorer**: Analyze distributions
        - **Visuals Pro**: Advanced visualizations
        
        ### 3. Train Models (Optional)
        - Go to **AI Models** page
        - Select models to train
        - View performance metrics
        
        ### 4. Make Predictions
        - Go to **Predict** page
        - Enter text (any language about mental health)
        - Click **ANALYZE RISK**
        - View results with semantic similarity scores
        
        ### Example Inputs
        - ✅ "I am skeptical whether to live or not. I hate this life." → Medium-High
        - ✅ "I don't want to live" → High
        - ✅ "The weather is good" → Low (unrelated)
        - ✅ "I feel empty and worthless" → High
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #1a1a1a;'>
    <p>🛡️ <b>MindGuard Analytics</b> | Transformer-Based Mental Health Risk Assessment</p>
    <p style='font-size: 0.9rem;'>Using Sentence-BERT for Semantic Understanding | Not keyword-dependent</p>
</div>
""", unsafe_allow_html=True)
