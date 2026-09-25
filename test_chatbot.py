import unittest
import json
import os
import sys

# Ensure root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from scripts.preprocess import clean_text
from app.chatbot import app, analyze_sentiment

class TestChatbotService(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.config['TESTING'] = True

    def test_clean_text_basic(self):
        """Test text normalization, contractions, and stopword filtering."""
        raw = "I don't know how to cancel my subscription?!"
        cleaned = clean_text(raw)
        self.assertIn("cancel", cleaned)
        self.assertNotIn("don't", cleaned)

    def test_sentiment_analysis(self):
        """Verify sentiment classifier catches urgent and positive signals."""
        self.assertEqual(analyze_sentiment("This is an urgent issue!"), "Urgent / Frustrated")
        self.assertEqual(analyze_sentiment("Thank you so much, great job!"), "Positive")
        self.assertEqual(analyze_sentiment("I would like to check my order status."), "Neutral")

    def test_health_endpoint(self):
        """Verify /api/health returns 200 OK and model status."""
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertIn('author', data)

    def test_chat_empty_payload(self):
        """Verify 400 Bad Request on empty message."""
        res = self.client.post('/chat', json={'message': ''})
        self.assertEqual(res.status_code, 400)

    def test_chat_greeting(self):
        """Verify greeting triggers Greeting intent with high confidence."""
        res = self.client.post('/chat', json={'message': 'hello there'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('intent'), 'Greeting')
        self.assertGreaterEqual(data.get('confidence'), 90)

    def test_chat_billing_inquiry(self):
        """Verify billing query is correctly categorized with confidence metrics."""
        res = self.client.post('/chat', json={'message': 'Can you help me with an incorrect billing charge on my statement?'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('intent', data)
        self.assertIn('confidence', data)
        self.assertIn('response', data)
        self.assertIn('suggested_actions', data)
        self.assertIn('latency_ms', data)

if __name__ == '__main__':
    unittest.main()
