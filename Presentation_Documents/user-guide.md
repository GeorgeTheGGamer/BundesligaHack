# User Guide: MoveNet Pose Recognition System

## Introduction

This guide will help you understand how to use our human pose recognition system, which uses Google's MoveNet pose estimation model combined with various machine learning classifiers to identify different human poses and movements.

## System Requirements

- Python 3.7+
- TensorFlow 2.x
- GPU is recommended for faster inference but not required

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movenet-pose-recognition.git
cd movenet-pose-recognition
```

2. Install required dependencies:
```bash
pip install tensorflow tensorflow-hub numpy opencv-python scikit-learn keras keras-tuner matplotlib pillow joblib
```

## Basic Usage

### 1. Image Classification

To classify a pose in an image:

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import joblib
from PIL import Image

# Load MoveNet
model_name = "movenet_thunder"
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

# Load one of the trained models (select based on your needs)
model = joblib.load('smart_ball_mlp.joblib')  # For ball-related poses
# model = joblib.load('smart_fullbody_mlp.joblib')  # For full-body poses
# model = joblib.load('smart_halfbody_mlp.joblib')  # For half-body poses
# model = joblib.load('naive_mlpclassifier.joblib')  # For general classification

# Function to process an image and get predictions
def classify_pose(image_path):
    # Load and prepare image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Get keypoints
    input_image = tf.expand_dims(img_array, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    model_fn = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model_fn(input_image)
    keypoints = outputs['output_0'].numpy()
    
    # Reshape for classifier input
    keypoints = keypoints.squeeze()
    keypoints = keypoints.reshape(1, 51)  # 17 keypoints × 3 values
    
    # Predict
    prediction = model.predict(keypoints)
    return prediction[0]

# Example usage
image_path = "path/to/your/image.jpg"
predicted_class = classify_pose(image_path)
print(f"Predicted pose class: {predicted_class}")
```

### 2. Using the Specialized Models

Our system uses a "smart" approach with specialized models for different types of poses:

#### Ball Poses (3 classes)
```python
model = joblib.load('smart_ball_mlp.joblib')
# Class 0: No Ball
# Class 1: Ball Held
# Class 2: Ball Thrown
```

#### Full Body Poses (10 classes)
```python
model = joblib.load('smart_fullbody_mlp.joblib')
# Classes 0-9: Various full-body poses
```

#### Half Body Poses (2 classes)
```python
model = joblib.load('smart_halfbody_mlp.joblib')
# Class 0: Upper body pose
# Class 1: Lower body pose
```

### 3. Using the Naive Model (All 16 classes)
```python
model = joblib.load('naive_mlpclassifier.joblib')
# Classes 0-15: All pose types combined
```

## Advanced Usage

### Real-time Classification with Webcam

```python
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import joblib

# Load MoveNet
model_name = "movenet_thunder"
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

# Load model
model = joblib.load('smart_fullbody_mlp.joblib')  # Choose appropriate model

# Function to process frame and get keypoints
def process_frame(frame):
    # Resize and pad frame
    input_image = tf.expand_dims(frame, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    
    # Get keypoints
    model_fn = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model_fn(input_image)
    keypoints = outputs['output_0'].numpy()
    
    # Reshape for classifier
    keypoints = keypoints.squeeze()
    keypoints_flat = keypoints.reshape(1, 51)
    
    # Get prediction
    prediction = model.predict(keypoints_flat)
    
    # Draw keypoints on frame
    draw_keypoints(frame, keypoints)
    
    return frame, prediction[0]

# Function to draw keypoints
def draw_keypoints(frame, keypoints):
    height, width, _ = frame.shape
    for i in range(17):
        x = int(keypoints[i, 1] * width)
        y = int(keypoints[i, 0] * height)
        conf = keypoints[i, 2]
        if conf > 0.3:  # Only draw high-confidence keypoints
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process frame
    processed_frame, prediction = process_frame(frame)
    
    # Display class on frame
    cv2.putText(processed_frame, f"Class: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('MoveNet Pose Classification', processed_frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### Common Issues

1. **Low confidence scores**: If you're getting inconsistent results, ensure:
   - The person is fully visible in the frame
   - There is adequate lighting
   - The person is not moving too quickly

2. **Model loading errors**: Make sure you have the correct path to your model files.

3. **Poor classification accuracy**: Try using the specialized models for better accuracy.

### Performance Optimization

- Use the Thunder variant of MoveNet for better accuracy, or Lightning for faster performance
- Reduce resolution for faster processing if needed
- Consider using GPU acceleration for real-time applications

## Extending the System

### Training on New Poses

To train the model on your custom poses:

1. Create folders for each pose class
2. Collect 20+ images per class
3. Use the provided code to extract keypoints and train models
4. Save your new models using joblib.dump()

### Integrating with Other Systems

The pose recognition system can be integrated with:
- Gaming applications
- Fitness tracking
- Movement analysis
- Interactive installations

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.
