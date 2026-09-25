#!/usr/bin/env python3
"""
CogniSupport - Production Launch Entrypoint
Engineered by Purushotham Balamurali
"""
import os
import sys

# Ensure root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.chatbot import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Launching CogniSupport on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
