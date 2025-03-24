from flask import Flask, render_template, request, jsonify
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from movenet import detect_pose
from objectdetection import detect_objects
import numpy as np

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(image_path):
    """Analyze an image to detect people and sports balls.
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        dict: Analysis results containing person and ball detection information
    """
    results = {
        'has_person': False,
        'person_type': None,
        'has_ball': False,
        'ball_confidence': 0.0
    }
    
    # Detect human pose using MoveNet
    try:
        keypoints = detect_pose(image_path)
        
        # If any keypoints are detected with confidence > 0.3, consider it a person
        if np.any(keypoints[:, 2] > 0.3):
            results['has_person'] = True
            
            # Determine if it's a full body, half body, or headshot
            # Keypoints order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
            
            # Check if head keypoints are visible (nose, eyes, ears)
            head_visible = np.any(keypoints[0:5, 2] > 0.5)  # Nose, eyes, ears
            
            # Check if torso keypoints are visible (shoulders, elbows, wrists, hips)
            torso_visible = np.any(keypoints[5:11, 2] > 0.3)  # Shoulders, elbows, wrists, hips
            
            # Check if leg keypoints are visible (knees, ankles)
            legs_visible = np.any(keypoints[11:17, 2] > 0.3)  # Knees, ankles
            
            if head_visible and torso_visible and legs_visible:
                results['person_type'] = 'Full body'
            elif head_visible and torso_visible:
                results['person_type'] = 'Half body'
            elif head_visible:
                results['person_type'] = 'Headshot'
            else:
                results['person_type'] = 'Person detected'
    except Exception as e:
        print(f"Error during pose detection: {e}")
    
    # Detect sports ball using object detection
    try:
        # Class ID 37 corresponds to "sports ball" in COCO dataset
        ball_score = detect_objects(image_path, class_id=37)
        
        if ball_score > 0.5:  # Consider balls detected with confidence > 0.5
            results['has_ball'] = True
            results['ball_confidence'] = float(ball_score)
    except Exception as e:
        print(f"Error during object detection: {e}")
    
    return results

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    return "Use GET method to access the upload page"

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        filepath = os.path.join(os.getcwd(), UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(filepath)
        
        # Analyze the image
        analysis_results = analyze_image(filepath)
        
        # Create a human-readable result message
        result_message = ""
        if analysis_results['has_person']:
            result_message += f"Person detected: {analysis_results['person_type']}. "
        else:
            result_message += "No person detected. "
            
        if analysis_results['has_ball']:
            result_message += f"Ball detected with {analysis_results['ball_confidence']:.1%} confidence."
        else:
            result_message += "No ball detected."
        
        # Return results
        return jsonify({
            "filename": filename,
            "message": result_message,
            "analysis": analysis_results
        })
    
    return jsonify({"error": "File type not allowed"}), 400

if __name__ == "__main__":
    app.run(debug=True)