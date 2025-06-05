import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uuid
import logging
import os
import pandas as pd
import altair as alt
import datetime
import random
from PIL import Image
import io
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="Swisbi - A Bot for Customer reviews",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - IMPROVED for better UI alignment
st.markdown("""
<style>
    /* Global font settings */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        color: #4B89DC;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        padding-top: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #3A3A3A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        padding-left: 0.5rem;
    }
    
    /* Chat bubbles - IMPROVED alignment */
    .user-bubble {
        background-color: #E8F4FC;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        display: inline-block;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
        color: #2C3E50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    .bot-bubble {
        background-color: #DCF8C6;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        display: inline-block;
        margin: 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
        color: #2C3E50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    
    /* Chat container - IMPROVED for better scrolling and responsiveness */
    .chat-container {
        padding: 1rem;
        border-radius: 12px;
        background-color: #f9f9f9;
        border: 1px solid #eaeaea;
        height: 450px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        min-height: 100%;
        width: 100%;
    }
    
    /* Message container - IMPROVED to fix alignment issues */
    .message-container {
        width: 100%;
        overflow: hidden;
        margin: 4px 0;
    }
    
    /* Cards styling - IMPROVED with consistent padding and margins */
    .mood-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .mood-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12);
    }
    
    .mood-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    
    .mood-label {
        font-size: 1.5rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Mood colors */
    .positive-mood {
        color: #2ECC71;
    }
    
    .neutral-mood {
        color: #F39C12;
    }
    
    .negative-mood {
        color: #E74C3C;
    }
    
    /* Button styling - IMPROVED for better alignment and hover effects */
    .stButton>button {
        background-color: #4B89DC;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        border: none;
        width: 100%;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #3A78CB;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    /* Input styling - IMPROVED for better alignment */
    .stTextInput>div>div>input {
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        border: 1px solid #d0d0d0;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Stats cards - IMPROVED layout and hover effects */
    .stat-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #555;
    }
    
    /* Sidebar content - IMPROVED padding and spacing */
    .sidebar-content {
        padding: 1.5rem 1rem;
        height: 100%;
    }
    
    /* Tips card - IMPROVED with better styling */
    .tips-card {
        background-color: #F0F7FF;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
        border-left: 4px solid #4B89DC;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Progress bar - IMPROVED colors and height */
    .stProgress > div > div > div > div {
        background-color: #4B89DC;
        height: 12px !important;
    }
    
    /* Progress bar container */
    .stProgress > div > div {
        background-color: #f0f0f0;
        border-radius: 8px !important;
        height: 12px !important;
    }
    
    /* Page header alignment fix */
    .stApp header {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Add feedback button style */
    .feedback-button {
        padding: 10px 15px;
        border-radius: 12px;
        border: none;
        font-weight: 500;
        margin: 5px;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .feedback-button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .positive-button {
        background-color: #2ECC71;
        color: white;
    }
    
    .neutral-button {
        background-color: #F39C12;
        color: white;
    }
    
    .negative-button {
        background-color: #E74C3C;
        color: white;
    }
    
    /* Responsive adjustments for mobile */
    @media (max-width: 768px) {
        .chat-container {
            height: 350px;
        }
        
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.3rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        st.error(f"Failed to download NLTK data: {str(e)}. Some features may not work.")

download_nltk_data()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = []
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'mood_tips' not in st.session_state:
    st.session_state.mood_tips = []
if 'username' not in st.session_state:
    st.session_state.username = "Friend"

# Model paths
MODEL_PATH = 'models/best_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'

# Parameters
max_len = 100

# Emoji definitions for moods
MOOD_EMOJIS = {
    'positive': '😄',
    'neutral': '😐',
    'negative': '😔'
}

# Check if model and tokenizer files exist
model_available = os.path.exists(MODEL_PATH)
tokenizer_available = os.path.exists(TOKENIZER_PATH)

# Show loading spinner
with st.spinner("Setting up Swisbi..."):
    # Try to load the model and tokenizer
    model = None
    tokenizer = None
    if model_available and tokenizer_available:
        try:
            model = load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded successfully")
            logger.info(f"Model output shape: {model.output_shape}")
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Model or tokenizer loading error: {str(e)}")
    
    # Use mock model if needed
    if model is None or tokenizer is None:
        logger.info("Using mock model due to missing or invalid files")
        class MockModel:
            def predict(self, text, **kwargs):
                # Simulate more realistic predictions
                return np.random.uniform(0, 1, size=(1, 3))  # Simulate 3-class output
                
        class MockTokenizer:
            def texts_to_sequences(self, text):
                return [[1, 2, 3]]
                
        model = MockModel()
        tokenizer = MockTokenizer()
        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    else:
        try:
            num_classes = model.output_shape[-1]
            if num_classes == 1:
                sentiment_mapping = {0: 'negative', 1: 'positive'}
                logger.info("Using binary sentiment mapping")
            else:
                sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
                logger.info(f"Using multi-class sentiment mapping with {num_classes} classes")
        except AttributeError:
            sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            logger.info("Using default multi-class sentiment mapping")
    
    reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
    time.sleep(0.5)

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Text preprocessing error: {str(e)}")
        return text

# Sentiment analysis (fixed)
def analyze_sentiment(text):
    if not text or not text.strip():
        logger.warning("Empty text received, returning neutral sentiment")
        return 'neutral', 0.5, 50
    
    processed = preprocess_text(text)
    if not processed:
        logger.warning("Empty processed text, returning default sentiment")
        return 'neutral', 0.5, 50
    
    # Expanded positive/negative word lists for better rule-based fallback
    positive_words = ['happy', 'good', 'great', 'excellent', 'wonderful', 'joy', 'love', 'excited', 
                     'amazing', 'awesome', 'fantastic', 'pleasant', 'glad', 'delighted', 'cheerful', 
                     'optimistic', 'thrilled', 'content', 'grateful']
    negative_words = ['sad', 'bad', 'awful', 'terrible', 'horrible', 'hate', 'angry', 'upset', 
                     'disappointed', 'worried', 'anxious', 'depressed', 'frustrated', 'unhappy', 
                     'miserable', 'stressed', 'lonely', 'gloomy']
    
    # Try model prediction
    try:
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        prediction = model.predict(padded, verbose=0)[0]
        
        # Handle different model output shapes
        num_classes = len(prediction) if hasattr(prediction, '__len__') else 1
        
        if num_classes == 1:
            # Binary classification
            prediction_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction
            pred_class = 1 if prediction_value > 0.5 else 0
            confidence = prediction_value if pred_class == 1 else 1 - prediction_value
        else:
            # Multi-class classification
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
        
        # Adjust confidence threshold for rule-based fallback
        if confidence < 0.5:  # Lowered threshold for more sensitivity
            text_lower = text.lower()
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)
            
            if pos_count > neg_count:
                pred_class = reverse_mapping.get('positive', 2)
                confidence = 0.7 + (0.2 * (pos_count - neg_count) / max(1, pos_count + neg_count))
            elif neg_count > pos_count:
                pred_class = reverse_mapping.get('negative', 0)
                confidence = 0.7 + (0.2 * (neg_count - pos_count) / max(1, pos_count + neg_count))
            else:
                pred_class = reverse_mapping.get('neutral', 1)
                confidence = 0.5
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        # Fallback to rule-based sentiment analysis
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            pred_class = reverse_mapping.get('positive', 2)
            confidence = 0.7 + (0.3 * min(1, pos_count / max(1, pos_count + neg_count)))
        elif neg_count > pos_count:
            pred_class = reverse_mapping.get('negative', 0)
            confidence = 0.7 + (0.3 * min(1, neg_count / max(1, pos_count + neg_count)))
        else:
            pred_class = reverse_mapping.get('neutral', 1)
            confidence = 0.5
    
    # Get sentiment label
    sentiment = sentiment_mapping.get(pred_class, 'neutral')
    
    # Fixed mood score calculation
    if sentiment == 'positive':
        mood_score = 60 + (confidence * 40)  # Adjusted to ensure positive scores > 60
    elif sentiment == 'negative':
        mood_score = 40 - (confidence * 40)  # Adjusted to ensure negative scores < 40
    else:
        mood_score = 50 + (confidence - 0.5) * 20  # Neutral scores around 50
    
    mood_score = round(min(max(mood_score, 0), 100))
    
    logger.info(f"Text: {text}")
    logger.info(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}, Mood Score: {mood_score}")
    return sentiment, confidence, mood_score

# Generate personalized mood improvement tips
def generate_mood_tips(sentiment, sentiment_history):
    positive_tips = [
        "Keep up that positive energy! Try sharing your happiness with someone else today.",
        "Your positivity is wonderful! Consider journaling about what's making you happy.",
        "Great mood! This is the perfect time to tackle a challenging task you've been putting off.",
        "Your positive mood can boost creativity. Why not try a creative activity today?",
        "Happiness increases when shared. Consider calling a friend to share your good mood."
    ]
    
    negative_tips = [
        "Take a few deep breaths. Inhale for 4 counts, hold for 4, and exhale for 6.",
        "Consider going for a short walk outside - even 10 minutes of fresh air can help.",
        "Try the 5-4-3-2-1 technique: Name 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste.",
        "Small wins matter. Try completing one simple task to build positive momentum.",
        "Your feelings are valid. Consider writing them down to process your emotions."
    ]
    
    neutral_tips = [
        "Consider setting a small, achievable goal for today.",
        "Mindfulness can help maintain emotional balance. Try focusing on your breathing for a few minutes.",
        "This balanced state is perfect for reflection. What's one thing you're grateful for today?",
        "A neutral mood is a good time to plan ahead. Consider what you'd like to accomplish tomorrow.",
        "Try engaging in an activity that usually brings you joy to elevate your mood even further."
    ]
    
    if len(sentiment_history) >= 3:
        recent_mood = sentiment_history[-3:]
        if all(s == 'negative' for s in recent_mood):
            return random.choice([
                "I've noticed you've been feeling down for a while. Remember it's okay to reach out for support from friends, family, or professionals.",
                "You've had several negative responses. If you're experiencing prolonged low mood, consider talking to a mental health professional."
            ])
        elif recent_mood[-2:] == ['negative', 'negative'] and recent_mood[0] == 'positive':
            return "I notice your mood has declined. What's changed since we started talking?"
        elif recent_mood[-2:] == ['positive', 'positive'] and recent_mood[0] == 'negative':
            return "Your mood seems to be improving! What's helping you feel better?"
    
    if sentiment == 'positive':
        return random.choice(positive_tips)
    elif sentiment == 'negative':
        return random.choice(negative_tips)
    else:
        return random.choice(neutral_tips)

# Generate chatbot response
def generate_response(user_input, sentiment, confidence, username="Friend"):
    responses = {
        'positive': [
            f"You sound really happy, {username}! That's wonderful to see.",
            f"I love your positive energy! What's contributing to your good mood?",
            f"That's great to hear! Your positivity is refreshing.",
            f"You seem to be in a great mood. Is something special happening?"
        ],
        'negative': [
            f"I'm sorry to hear you're feeling down, {username}. Would you like to talk about it?",
            f"It sounds like things are difficult right now. Remember that it's okay to have tough days.",
            f"I'm here to listen if you want to share more about what's troubling you."
        ],
        'neutral': [
            f"I sense a balanced perspective in your message, {username}.",
            f"You seem to be taking things in stride. How are you feeling overall?",
            f"Thanks for sharing that. Is there anything specific on your mind today?"
        ]
    }
    
    lower_input = user_input.lower()
    if any(word in lower_input for word in ['hello', 'hi', 'hey', 'greetings']):
        return f"Hello {username}! How are you feeling today? I'm here to chat and track your mood."
    if "what can you do" in lower_input or "how do you work" in lower_input:
        return f"I'm Swisbi, your emotional analysis companion! I can analyze your mood based on our conversation, track mood patterns over time, and offer personalized suggestions. Just chat with me naturally and I'll do the rest!"
    if any(word in lower_input for word in ['thanks', 'thank you', 'appreciate']):
        return f"You're very welcome, {username}! I'm happy to be here for you."
    if any(word in lower_input for word in ['bye', 'goodbye', 'see you', 'farewell']):
        return f"It was nice chatting with you, {username}! Take care and come back anytime."
    
    if sentiment not in responses:
        sentiment = 'neutral'
    
    base_response = random.choice(responses[sentiment])
    
    if len(st.session_state.sentiment_history) > 3:
        recent_sentiments = st.session_state.sentiment_history[-3:]
        if all(s == 'positive' for s in recent_sentiments):
            follow_up = f" You've been in a great mood throughout our conversation. What's contributing to this positive streak?"
        elif all(s == 'negative' for s in recent_sentiments):
            follow_up = f" I've noticed you've been feeling down for a while. Would talking about it help? Remember that it's okay to seek support."
        elif recent_sentiments[-1] != recent_sentiments[-2]:
            prev_mood = recent_sentiments[-2]
            curr_mood = recent_sentiments[-1]
            follow_up = f" I notice your mood has shifted from {prev_mood} to {curr_mood}. Did something specific trigger this change?"
        else:
            follow_up = ""
    else:
        follow_up = ""
    
    if len(st.session_state.sentiment_history) % 3 == 0 and len(st.session_state.sentiment_history) > 0:
        tip = generate_mood_tips(sentiment, st.session_state.sentiment_history)
        st.session_state.mood_tips.append(tip)
        follow_up = f"{follow_up}\n\n💡 *Mood Tip: {tip}*"
    
    return base_response + follow_up

# Helper function to get color based on sentiment
def get_sentiment_color(sentiment):
    if sentiment == 'positive':
        return "#2ECC71"
    elif sentiment == 'negative':
        return "#E74C3C"
    else:
        return "#F39C12"

# Helper function to create mood chart
def create_mood_chart(scores, sentiments, timestamps):
    if not scores or len(scores) < 2:
        return None
    
    chart_data = pd.DataFrame({
        'Timestamp': timestamps,
        'Score': scores,
        'Sentiment': sentiments
    })
    
    chart_data['Timestamp'] = pd.to_datetime(chart_data['Timestamp'])
    chart_data['Time'] = chart_data['Timestamp'].dt.strftime('%H:%M')
    
    # Improved chart with better styling
    chart = alt.Chart(chart_data).mark_line(
        point=alt.OverlayMarkDef(size=100, filled=True)
    ).encode(
        x=alt.X('Timestamp:T', 
                title='Time', 
                axis=alt.Axis(format='%H:%M', labelAngle=-45, grid=False)),
        y=alt.Y('Score:Q', 
                scale=alt.Scale(domain=[0, 100], nice=True), 
                title='Mood Score',
                axis=alt.Axis(grid=True)),
        color=alt.Color('Sentiment:N', 
                        scale=alt.Scale(domain=['positive', 'neutral', 'negative'],
                                        range=['#2ECC71', '#F39C12', '#E74C3C']),
                        legend=alt.Legend(title="Mood", orient='top')),
        tooltip=['Time', 'Score', 'Sentiment']
    ).properties(
        width='container',
        height=300,
        title='Your Mood Over Time'
    ).configure_title(
        fontSize=20,
        font='Arial',
        anchor='start',
        color='#333'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelFont='Arial',
        titleFont='Arial',
        labelFontSize=12,
        titleFontSize=14
    )
    
    return chart

# Helper function to format chat message - IMPROVED for better alignment
def format_message(role, content):
    if role == "user":
        return f'<div class="message-container"><div class="user-bubble">{content}</div></div>'
    else:
        return f'<div class="message-container"><div class="bot-bubble">{content}</div></div>'

# Function to calculate mood statistics
def calculate_mood_stats():
    if not st.session_state.sentiment_history:
        return {"positive": 0, "neutral": 0, "negative": 0, "avg_score": 0}
    
    sentiment_counts = {
        "positive": st.session_state.sentiment_history.count("positive"),
        "neutral": st.session_state.sentiment_history.count("neutral"),
        "negative": st.session_state.sentiment_history.count("negative")
    }
    
    avg_score = sum(st.session_state.sentiment_scores) / len(st.session_state.sentiment_scores) if st.session_state.sentiment_scores else 0
    
    return {**sentiment_counts, "avg_score": round(avg_score, 1)}

# Function to display mood tips - IMPROVED styling

    
        

# Function to provide user feedback - IMPROVED UI
def display_feedback_section():
    st.markdown('<h3 class="sub-header">Feedback</h3>', unsafe_allow_html=True)
    
    if not st.session_state.feedback_given:
        st.markdown("<p>How is your experience with Swisbi today?</p>", unsafe_allow_html=True)
        
        # Using custom HTML buttons for better styling
        feedback_html = """
        <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
            <button class="feedback-button positive-button" onclick="document.getElementById('positive-feedback').click()">😄 Great</button>
            <button class="feedback-button neutral-button" onclick="document.getElementById('neutral-feedback').click()">😐 Okay</button>
            <button class="feedback-button negative-button" onclick="document.getElementById('negative-feedback').click()">😔 Poor</button>
        </div>
        """
        st.markdown(feedback_html, unsafe_allow_html=True)
        
        # Hidden buttons that will be clicked by the HTML buttons
        # Hidden buttons that will be clicked by the HTML buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Great", key="positive-feedback", type="primary", use_container_width=True):
                st.session_state.feedback_given = True
                st.rerun()
        with col2:
            if st.button("Okay", key="neutral-feedback", type="primary", use_container_width=True):
                st.session_state.feedback_given = True
                st.rerun()
        with col3:
            if st.button("Poor", key="negative-feedback", type="primary", use_container_width=True):
                st.session_state.feedback_given = True
                st.rerun()
    else:
        st.success("Thank you for your feedback! We're always working to improve your experience.")

# Main layout with improved organization
def display_main_app():
    # Initialize username or use default
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    if st.session_state.username == "Friend":
        new_username = st.sidebar.text_input("Enter your name:", value="", placeholder="Your name")
        if new_username:
            st.session_state.username = new_username
    else:
        st.sidebar.markdown(f"<h3>Hello, {st.session_state.username}!</h3>", unsafe_allow_html=True)
        if st.sidebar.button("Change name"):
            st.session_state.username = "Friend"
            st.rerun()
    
    # Display mood statistics in sidebar
    st.sidebar.markdown('<h3 class="sub-header">Your Review Stats</h3>', unsafe_allow_html=True)
    
    stats = calculate_mood_stats()
    total_messages = len(st.session_state.sentiment_history)
    
    # Calculate percentages for display
    pos_percent = stats["positive"] / total_messages * 100 if total_messages > 0 else 0
    neu_percent = stats["neutral"] / total_messages * 100 if total_messages > 0 else 0
    neg_percent = stats["negative"] / total_messages * 100 if total_messages > 0 else 0
    
    # Progress bars for mood distribution
    st.sidebar.markdown("**Review Distribution:**")
    
    st.sidebar.markdown(f"Positive {MOOD_EMOJIS['positive']}")
    st.sidebar.progress(pos_percent / 100)
    st.sidebar.markdown(f"Neutral {MOOD_EMOJIS['neutral']}")
    st.sidebar.progress(neu_percent / 100)
    st.sidebar.markdown(f"Negative {MOOD_EMOJIS['negative']}")
    st.sidebar.progress(neg_percent / 100)
    
    # Display Mood Tips
    
    # Feedback section
    display_feedback_section()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    st.markdown('<h1 class="main-header">Swisbi Chatbot</h1>', unsafe_allow_html=True)
    
    # Display current mood if we have messages
    if st.session_state.sentiment_scores:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            latest_score = st.session_state.sentiment_scores[-1]
            latest_sentiment = st.session_state.sentiment_history[-1]
            
            mood_class = ''
            if latest_sentiment == 'positive':
                mood_class = 'positive-mood'
            elif latest_sentiment == 'negative':
                mood_class = 'negative-mood'
            else:
                mood_class = 'neutral-mood'
            
            st.markdown(f"""
            <div class="mood-card">
                <div class="mood-score {mood_class}">{MOOD_EMOJIS[latest_sentiment]} {latest_score}</div>
                <div class="mood-label">Current Mood</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display mood chart if we have enough data points
    if len(st.session_state.sentiment_scores) >= 2:
        st.markdown('<h3 class="sub-header">Mood Tracker</h3>', unsafe_allow_html=True)
        chart = create_mood_chart(
            st.session_state.sentiment_scores,
            st.session_state.sentiment_history,
            st.session_state.timestamps
        )
        if chart:
            st.altair_chart(chart, use_container_width=True)
    
    # Chat interface
    st.markdown('<h3 class="sub-header">Chat with Swisbi</h3>', unsafe_allow_html=True)
    
    # Display chat container with proper styling
    chat_container = st.container()
    with chat_container:
        
        # Add welcome message if first visit
        if not st.session_state.welcome_shown:
            welcome_msg = f"Hello {st.session_state.username}! I'm Swisbi, your Customer review analysis companion. What is your review today?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.session_state.welcome_shown = True
        
        # Display all messages
        for message in st.session_state.messages:
            st.markdown(format_message(message["role"], message["content"]), unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # User input
    user_input = st.text_input(
        "Type your message:",
        key=f"user_input_{st.session_state.input_key}",
        on_change=None,
        placeholder="Share your review..."
    )
        
    # Process user input
    if st.button("Send", use_container_width=True):
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Analyze sentiment and generate response
            sentiment, confidence, mood_score = analyze_sentiment(user_input)
            
            # Store results
            st.session_state.sentiment_history.append(sentiment)
            st.session_state.sentiment_scores.append(mood_score)
            st.session_state.timestamps.append(datetime.datetime.now())
            
            # Generate and add bot response
            bot_response = generate_response(user_input, sentiment, confidence, st.session_state.username)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
            # Reset input and trigger a rerun to update the UI
            st.session_state.input_key += 1
            st.rerun()

# Display the app
display_main_app()

# Option to reset the conversation
with st.expander("Reset Options"):
    if st.button("Start New Conversation"):
        # Reset all conversation-related state
        st.session_state.messages = []
        st.session_state.sentiment_history = []
        st.session_state.sentiment_scores = []
        st.session_state.timestamps = []
        st.session_state.welcome_shown = False
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.input_key = 0
        st.session_state.feedback_given = False
        st.session_state.mood_tips = []
        st.rerun()
