import os
import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import sqlite3
import hashlib
import secrets
from functools import wraps
import time

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for frontend integration
CORS(app, origins=['http://localhost:8000', 'https://yourdomain.com'])

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    def __init__(self):
        self.model_path = os.environ.get('MODEL_PATH', 'models/mask_detector.h5')
        self.face_cascade_path = os.environ.get('FACE_CASCADE_PATH', 
                                               cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.5'))
        self.database_path = os.environ.get('DATABASE_PATH', 'detection_logs.db')
        self.debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

config = Config()

class DatabaseManager:
    """Handle all database operations"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Detection logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        image_hash TEXT,
                        faces_detected INTEGER,
                        masked_faces INTEGER,
                        unmasked_faces INTEGER,
                        confidence_scores TEXT,
                        processing_time REAL,
                        detection_method TEXT,
                        user_ip TEXT,
                        session_id TEXT
                    )
                ''')
                
                # Statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE DEFAULT CURRENT_DATE,
                        total_detections INTEGER DEFAULT 0,
                        total_masked INTEGER DEFAULT 0,
                        total_unmasked INTEGER DEFAULT 0,
                        compliance_rate REAL DEFAULT 0.0,
                        avg_processing_time REAL DEFAULT 0.0
                    )
                ''')
                
                # API usage table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS api_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        endpoint TEXT,
                        user_ip TEXT,
                        response_time REAL,
                        status_code INTEGER
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def log_detection(self, detection_data):
        """Log detection results to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detection_logs 
                    (image_hash, faces_detected, masked_faces, unmasked_faces, 
                     confidence_scores, processing_time, detection_method, user_ip, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', detection_data)
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log detection: {str(e)}")
            return None
    
    def get_statistics(self, days=7):
        """Get detection statistics for specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_detections,
                        SUM(masked_faces) as total_masked,
                        SUM(unmasked_faces) as total_unmasked,
                        AVG(processing_time) as avg_processing_time,
                        DATE(timestamp) as date
                    FROM detection_logs 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                '''.format(days))
                
                stats = cursor.fetchall()
                
                # Calculate overall compliance rate
                total_faces = sum(row[1] + row[2] for row in stats if row[1] and row[2])
                total_masked = sum(row[1] for row in stats if row[1])
                compliance_rate = (total_masked / total_faces * 100) if total_faces > 0 else 0
                
                return {
                    'daily_stats': [
                        {
                            'date': row[4],
                            'total_detections': row[0],
                            'masked': row[1] or 0,
                            'unmasked': row[2] or 0,
                            'avg_processing_time': round(row[3] or 0, 3)
                        } for row in stats
                    ],
                    'overall_compliance_rate': round(compliance_rate, 2),
                    'total_detections': sum(row[0] for row in stats),
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return None

# Initialize database
db_manager = DatabaseManager(config.database_path)

class FaceMaskDetector:
    """Core face mask detection class"""
    
    def __init__(self):
        self.face_cascade = None
        self.mask_model = None
        self.load_models()
    
    def load_models(self):
        """Load face detection and mask classification models"""
        try:
            # Load OpenCV face cascade
            if os.path.exists(config.face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(config.face_cascade_path)
                logger.info("Face cascade loaded successfully")
            else:
                logger.warning("Face cascade file not found, using alternative method")
            
            # Load mask detection model if available
            if os.path.exists(config.model_path):
                self.mask_model = load_model(config.model_path)
                logger.info("Mask detection model loaded successfully")
            else:
                logger.warning("Mask detection model not found, using heuristic method")
                
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
    
    def detect_faces(self, image):
        """Detect faces in image using OpenCV"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                return faces
            else:
                # Alternative face detection using contours (basic)
                return self.detect_faces_alternative(gray)
                
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def detect_faces_alternative(self, gray_image):
        """Alternative face detection method"""
        # Basic skin color detection in grayscale
        # This is a simplified approach - in production, use proper face detection
        faces = []
        height, width = gray_image.shape
        
        # Simple blob detection for face-like regions
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum face size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.3:  # Face-like aspect ratio
                    faces.append([x, y, w, h])
        
        return np.array(faces)
    
    def classify_mask(self, face_image):
        """Classify if face is wearing a mask"""
        try:
            if self.mask_model is not None:
                # Use trained model
                face_resized = cv2.resize(face_image, (224, 224))
                face_array = img_to_array(face_resized)
                face_array = np.expand_dims(face_array, axis=0)
                face_array = face_array / 255.0
                
                predictions = self.mask_model.predict(face_array)
                mask_prob = predictions[0][0]  # Assuming model outputs [mask_prob, no_mask_prob]
                
                return {
                    'has_mask': mask_prob > config.confidence_threshold,
                    'confidence': float(mask_prob),
                    'method': 'neural_network'
                }
            else:
                # Use heuristic method
                return self.classify_mask_heuristic(face_image)
                
        except Exception as e:
            logger.error(f"Mask classification failed: {str(e)}")
            return {'has_mask': False, 'confidence': 0.0, 'method': 'error'}
    
    def classify_mask_heuristic(self, face_image):
        """Heuristic mask detection based on color analysis"""
        try:
            # Focus on lower half of face where mask would be
            height, width = face_image.shape[:2]
            lower_face = face_image[int(height * 0.6):, :]
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(lower_face, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution
            avg_brightness = np.mean(lab[:, :, 0])
            color_variance = np.var(hsv[:, :, 1])
            
            # Mask indicators: lower brightness, higher color variance
            mask_score = 0.0
            
            if avg_brightness < 100:  # Darker region
                mask_score += 0.4
            
            if color_variance > 200:  # More colorful (non-skin)
                mask_score += 0.3
            
            # Edge detection for mask boundaries
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_lower, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1:  # High edge density
                mask_score += 0.3
            
            has_mask = mask_score > config.confidence_threshold
            
            return {
                'has_mask': has_mask,
                'confidence': min(mask_score, 1.0),
                'method': 'heuristic'
            }
            
        except Exception as e:
            logger.error(f"Heuristic mask detection failed: {str(e)}")
            return {'has_mask': False, 'confidence': 0.0, 'method': 'error'}
    
    def process_image(self, image):
        """Main image processing pipeline"""
        start_time = time.time()
        
        try:
            # Detect faces
            faces = self.detect_faces(image)
            
            results = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Classify mask
                mask_result = self.classify_mask(face_roi)
                
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'has_mask': mask_result['has_mask'],
                    'confidence': mask_result['confidence'],
                    'method': mask_result['method']
                })
            
            processing_time = time.time() - start_time
            
            return {
                'faces': results,
                'total_faces': len(results),
                'masked_faces': sum(1 for r in results if r['has_mask']),
                'unmasked_faces': sum(1 for r in results if not r['has_mask']),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return {
                'faces': [],
                'total_faces': 0,
                'masked_faces': 0,
                'unmasked_faces': 0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

# Initialize detector
detector = FaceMaskDetector()

# Utility functions
def validate_api_key():
    """Decorator to validate API key for protected endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            expected_key = os.environ.get('API_KEY')
            
            if expected_key and api_key != expected_key:
                return jsonify({'error': 'Invalid API key'}), 401
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_api_usage(endpoint, response_time, status_code):
    """Log API usage for monitoring"""
    try:
        with sqlite3.connect(config.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_usage (endpoint, user_ip, response_time, status_code)
                VALUES (?, ?, ?, ?)
            ''', (endpoint, get_remote_address(), response_time, status_code))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log API usage: {str(e)}")

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_data))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Base64 image decoding failed: {str(e)}")
        return None

# API Routes
@app.route('/')
def index():
    """Serve the main application"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'face_detection': self.face_cascade is not None,
            'mask_model': detector.mask_model is not None,
            'database': os.path.exists(config.database_path)
        }
    })

@app.route('/api/detect', methods=['POST'])
@limiter.limit("30 per minute")
def detect_masks():
    """Main mask detection endpoint"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Process image
        result = detector.process_image(image)
        
        # Generate image hash for logging
        image_hash = hashlib.md5(data['image'].encode()).hexdigest()
        
        # Log detection
        if 'error' not in result:
            detection_data = (
                image_hash,
                result['total_faces'],
                result['masked_faces'],
                result['unmasked_faces'],
                json.dumps([face['confidence'] for face in result['faces']]),
                result['processing_time'],
                'server_processing',
                get_remote_address(),
                request.headers.get('X-Session-ID', 'anonymous')
            )
            db_manager.log_detection(detection_data)
        
        # Log API usage
        response_time = time.time() - start_time
        log_api_usage('detect', response_time, 200)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection endpoint error: {str(e)}")
        log_api_usage('detect', time.time() - start_time, 500)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats', methods=['GET'])
@limiter.limit("10 per minute")
def get_statistics():
    """Get detection statistics"""
    try:
        days = request.args.get('days', 7, type=int)
        stats = db_manager.get_statistics(days)
        
        if stats is None:
            return jsonify({'error': 'Failed to retrieve statistics'}), 500
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Statistics endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/openai-detect', methods=['POST'])
@limiter.limit("5 per minute")
@validate_api_key()
def openai_detect():
    """Enhanced detection using OpenAI API"""
    if not config.openai_api_key:
        return jsonify({'error': 'OpenAI API not configured'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Prepare OpenAI request
        headers = {
            'Authorization': f'Bearer {config.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-4-vision-preview',
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Analyze this image for face mask compliance. Return JSON with detected faces, bounding boxes (x,y,width,height), and mask status (true/false) with confidence scores.'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': data['image']
                        }
                    }
                ]
            }],
            'max_tokens': 300
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            
            try:
                # Parse JSON response
                detection_result = json.loads(analysis)
                return jsonify({
                    'success': True,
                    'detections': detection_result,
                    'method': 'openai_gpt4_vision'
                })
            except json.JSONDecodeError:
                return jsonify({
                    'success': False,
                    'error': 'Failed to parse OpenAI response',
                    'raw_response': analysis
                })
        else:
            return jsonify({
                'error': f'OpenAI API error: {response.status_code}'
            }), response.status_code
            
    except Exception as e:
        logger.error(f"OpenAI detection error: {str(e)}")
        return jsonify({'error': 'OpenAI detection failed'}), 500

@app.route('/api/batch-detect', methods=['POST'])
@limiter.limit("5 per minute")
@validate_api_key()
def batch_detect():
    """Batch processing for multiple images"""
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400
        
        images = data['images']
        if len(images) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 images per batch'}), 400
        
        results = []
        for i, image_data in enumerate(images):
            try:
                image = decode_base64_image(image_data)
                if image is not None:
                    result = detector.process_image(image)
                    result['image_index'] = i
                    results.append(result)
                else:
                    results.append({
                        'image_index': i,
                        'error': 'Invalid image format'
                    })
            except Exception as e:
                results.append({
                    'image_index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        return jsonify({'error': 'Batch processing failed'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    return jsonify({
        'face_detection': {
            'available': detector.face_cascade is not None,
            'method': 'opencv_cascade' if detector.face_cascade else 'alternative'
        },
        'mask_classification': {
            'available': detector.mask_model is not None,
            'method': 'neural_network' if detector.mask_model else 'heuristic'
        },
        'confidence_threshold': config.confidence_threshold,
        'supported_formats': ['JPEG', 'PNG', 'WebP']
    })

@app.route('/api/export-data', methods=['GET'])
@limiter.limit("2 per hour")
@validate_api_key()
def export_data():
    """Export detection data as CSV"""
    try:
        days = request.args.get('days', 30, type=int)
        format_type = request.args.get('format', 'csv')
        
        with sqlite3.connect(config.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, faces_detected, masked_faces, unmasked_faces,
                       processing_time, detection_method, user_ip
                FROM detection_logs
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days))
            
            data = cursor.fetchall()
            
            if format_type == 'json':
                result = [{
                    'timestamp': row[0],
                    'faces_detected': row[1],
                    'masked_faces': row[2],
                    'unmasked_faces': row[3],
                    'processing_time': row[4],
                    'detection_method': row[5],
                    'user_ip': row[6]
                } for row in data]
                
                return jsonify({
                    'data': result,
                    'total_records': len(result),
                    'period_days': days
                })
            
            else:  # CSV format
                import csv
                output = BytesIO()
                output_str = BytesIO()
                
                # Write CSV data
                fieldnames = ['timestamp', 'faces_detected', 'masked_faces', 
                             'unmasked_faces', 'processing_time', 'detection_method', 'user_ip']
                
                csv_content = "timestamp,faces_detected,masked_faces,unmasked_faces,processing_time,detection_method,user_ip\n"
                for row in data:
                    csv_content += ",".join(str(col) for col in row) + "\n"
                
                return csv_content, 200, {
                    'Content-Type': 'text/csv',
                    'Content-Disposition': f'attachment; filename=detection_data_{days}days.csv'
                }
        
    except Exception as e:
        logger.error(f"Data export error: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'retry_after': str(e.retry_after)}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = config.debug
    
    logger.info(f"Starting Face Mask Detection Server on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Database: {config.database_path}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
