import numpy as np
import os
import sys
import time
import re
import random
import joblib
from flask import Flask, request, jsonify, render_template

# Setup base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocess import clean_text

MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'intent_classifier.pkl')
VEC_PATH = os.path.join(PROJECT_ROOT, 'model', 'vectorizer.pkl')

# Load model & vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
except Exception as e:
    print(f"Warning: Loading model from fallback path. Error: {e}")
    model = None
    vectorizer = None

# Curated contextual response matrix
RESPONSE_MAP = {
    'Greeting': [
        "Hello! I am your AI Customer Support Assistant. How can I help you today?",
        "Hi there! Welcome to support. What can I assist you with today?",
        "Greetings! Feel free to ask about billing, cancellations, refunds, or technical issues."
    ],
    'Billing inquiry': [
        "I can assist with billing queries. You can review current invoices, update your payment card, or inspect disputed charges in the Billing Portal.",
        "Your billing statement reflects active subscriptions. Would you like a copy of your recent invoice or payment receipt?",
        "For billing updates, ensure your payment details are verified. Let me know if you need assistance with a specific invoice."
    ],
    'Cancellation request': [
        "I understand you are inquiring about cancellation. While we would love to keep you, I can initiate the cancellation workflow or review pause options.",
        "To process your cancellation, please confirm your registered account email and reason for cancellation so we can finalize the request.",
        "We are sorry to see you consider leaving. Before cancelling, would you like to review alternative plans or speak with a specialist?"
    ],
    'Product inquiry': [
        "Our product suite offers comprehensive enterprise tools, automated API integrations, and dedicated SLA support tiers.",
        "Here are our key capabilities: high-availability data pipelines, authenticated services, and automated workflow triggers.",
        "Would you like product documentation, pricing tier specifications, or a feature roadmap overview?"
    ],
    'Refund request': [
        "I can help process your refund request. Standard refunds are processed within 3-5 business days to your original payment method.",
        "To review your refund eligibility, please have your Order ID or Transaction Reference number ready.",
        "Your satisfaction is essential to us. Let me connect your request with our billing ledger to verify eligibility."
    ],
    'Technical issue': [
        "Let's troubleshoot that technical problem. Are you experiencing an API timeout, error code, or authentication failure?",
        "I can help resolve technical glitches. Please share the specific error message or steps to reproduce the issue.",
        "Our engineering systems are monitored 24/7. Let's isolate whether this is an environment configuration or endpoint issue."
    ],
    'Fallback': [
        "I want to make sure I get this right. Could you rephrase your question or select from one of the quick options below?",
        "I could not confidently match that request. Would you like me to route you to a human support specialist?",
        "I am not entirely certain about that inquiry. Please specify if your question is regarding Billing, Technical Support, or Refunds."
    ]
}

SUGGESTED_ACTIONS = {
    'Billing inquiry': ['View Latest Invoice', 'Update Payment Method', 'Dispute Charge'],
    'Cancellation request': ['Pause Subscription', 'Download Data Export', 'Confirm Cancellation'],
    'Product inquiry': ['Explore Feature Docs', 'Compare Pricing Plans', 'Request Demo'],
    'Refund request': ['Check Refund Status', 'Submit Refund Ticket', 'Policy Guidelines'],
    'Technical issue': ['System Status Page', 'API Error Codes', 'Report Bug'],
    'Greeting': ['Billing Question', 'Technical Issue', 'Refund Help'],
    'Fallback': ['Billing Inquiry', 'Technical Issue', 'Refund Help', 'Speak to Human']
}

def analyze_sentiment(text):
    """Rule-based sentiment and urgency detection."""
    text_lower = text.lower()
    urgent_keywords = ['urgent', 'immediately', 'asap', 'broken', 'horrible', 'frustrated', 'lawsuit', 'angry', 'scam', 'terrible', 'worst']
    positive_keywords = ['thank', 'great', 'awesome', 'helpful', 'appreciate', 'good', 'excellent', 'love']
    
    if any(k in text_lower for k in urgent_keywords):
        return 'Urgent / Frustrated'
    elif any(k in text_lower for k in positive_keywords):
        return 'Positive'
    return 'Neutral'

# Create Flask app
app = Flask(__name__, template_folder=os.path.join(CURRENT_DIR, 'templates'), static_folder=os.path.join(CURRENT_DIR, 'static'))

# In-memory session metrics
SESSION_METRICS = {
    'total_queries': 0,
    'intent_counts': {},
    'fallback_count': 0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'total_queries_processed': SESSION_METRICS['total_queries'],
        'author': 'Purushotham Balamurali'
    })

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    user_msg = data.get('message', '').strip()
    
    if not user_msg:
        return jsonify({'error': 'Message cannot be empty.'}), 400

    SESSION_METRICS['total_queries'] += 1

    # Check greeting heuristics
    greeting_triggers = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
    if user_msg.lower() in greeting_triggers or (len(user_msg.split()) <= 2 and any(g in user_msg.lower() for g in greeting_triggers)):
        predicted_intent = 'Greeting'
        confidence = 0.99
    elif model and vectorizer:
        cleaned = clean_text(user_msg)
        if not cleaned:
            cleaned = user_msg.lower()
        
        vec = vectorizer.transform([cleaned])
        predicted_intent = model.predict(vec)[0]
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(vec)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 0.85
    else:
        predicted_intent = 'Fallback'
        confidence = 0.0

    # Low-confidence thresholding guard
    CONFIDENCE_THRESHOLD = 0.35
    if confidence < CONFIDENCE_THRESHOLD and predicted_intent != 'Greeting':
        original_intent = predicted_intent
        predicted_intent = 'Fallback'
        SESSION_METRICS['fallback_count'] += 1
    else:
        original_intent = predicted_intent

    # Update metrics
    SESSION_METRICS['intent_counts'][predicted_intent] = SESSION_METRICS['intent_counts'].get(predicted_intent, 0) + 1

    # Select response
    responses = RESPONSE_MAP.get(predicted_intent, RESPONSE_MAP['Fallback'])
    selected_response = random.choice(responses)

    # Sentiment analysis
    sentiment = analyze_sentiment(user_msg)

    # Latency calculation
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'intent': predicted_intent,
        'original_intent': original_intent,
        'confidence': round(confidence * 100, 1),
        'confidence_raw': round(confidence, 4),
        'response': selected_response,
        'sentiment': sentiment,
        'suggested_actions': SUGGESTED_ACTIONS.get(predicted_intent, SUGGESTED_ACTIONS['Fallback']),
        'latency_ms': latency_ms
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
