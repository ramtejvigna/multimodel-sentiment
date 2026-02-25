"""
Flask Web Application for Multimodal Sentiment Analysis
Real-time text and facial emotion analysis with fusion
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image
import io
from pathlib import Path

from multimodal_inference import create_multimodal_analyzer

app = Flask(__name__)
CORS(app)

# Global analyzer instance
analyzer = None


def initialize_analyzer():
    """Initialize multimodal analyzer"""
    global analyzer
    
    try:
        print("\nInitializing multimodal analyzer...")
        analyzer = create_multimodal_analyzer(
            fusion_type='late',
            fusion_weights={'text_weight': 0.4, 'face_weight': 0.6}
        )
        print("Analyzer initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        print("\nMake sure you have trained models:")
        print("  - Face model: output/models/best_model.pth")
        print("  - Text model: output/text_models/best_text_model.pth")
        print("  - Vocabulary: output/text_models/vocabulary.pkl")
        return False


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text sentiment"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if analyzer is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        result = analyzer.analyze_text(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/face', methods=['POST'])
def analyze_face():
    """Analyze facial emotion from image"""
    try:
        if 'image' not in request.files:
            # Check for base64 image
            data = request.get_json()
            if data and 'image' in data:
                # Decode base64 image
                image_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(image_data))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            # File upload
            file = request.files['image']
            image = Image.open(file.stream)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if analyzer is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        result = analyzer.analyze_face(image)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/multimodal', methods=['POST'])
def analyze_multimodal():
    """Analyze both text and face"""
    try:
        # Get text
        text = request.form.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if analyzer is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        result = analyzer.analyze_multimodal(text, image, return_individual=True)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status"""
    return jsonify({
        'status': 'ready' if analyzer else 'not_initialized',
        'models_loaded': analyzer is not None
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis Web App')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multimodal Sentiment Analysis - Web Application")
    print("="*60)
    
    # Initialize analyzer
    if initialize_analyzer():
        print(f"\nStarting server on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        print("="*60)
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    else:
        print("\nFailed to start server: Models not initialized")
        print("Please train the models first.")
