# Human Pose Recognition with MoveNet

This project implements a human pose recognition system using Google's MoveNet pose estimation model combined with various machine learning classifiers. The system can identify different human poses and movements by analyzing skeletal keypoints extracted from images.

## Overview

The system uses the following pipeline:
1. Extract 17 keypoints from human images using MoveNet Thunder
2. Use these keypoints as features for machine learning models
3. Classify poses into different categories

## Project Structure

```
├── models/                  # Saved trained models
│   ├── NaiveKFoldModel.keras
│   ├── NaiveKFoldsTuned.keras
│   ├── SmartBallKFoldsModel.keras
│   ├── SmartBallKFoldsTuned.keras
│   ├── SmartFullbodyKFoldsModels.keras
│   ├── SmartFullBodyKFoldsTuned.keras
│   ├── SmartHalfBodyKfoldsModel.keras
│   └── SmartHalfBodyTuned.keras
├── naive_k_model.joblib     # Saved K-Nearest Neighbors model (naive)
├── naive_mlpclassifier.joblib  # Saved MLP classifier model (naive)
├── smart_ball_k_model.joblib    # Saved K-Nearest Neighbors model (ball)
├── smart_ball_mlp.joblib        # Saved MLP classifier model (ball)
├── smart_fullbody_k_model.joblib  # Saved K-Nearest model (fullbody)
├── smart_fullbody_mlp.joblib      # Saved MLP model (fullbody)
├── smart_halfbody_k_model.joblib  # Saved K-Nearest model (halfbody)
├── smart_halfbody_mlp.joblib      # Saved MLP model (halfbody)
├── TestImages/              # Test image dataset
├── MoveNetSmart/            # Smart approach image datasets
│   ├── ResizedImages/
│       ├── Ball/
│       ├── FullBody/
│       └── HalfBody/
└── paste.txt                # Original code file
```

## Requirements

- TensorFlow 2.x
- TensorFlow Hub
- NumPy
- OpenCV
- scikit-learn
- Keras
- Keras Tuner
- Matplotlib
- PIL (Pillow)
- Joblib

## Approach

The project explores two main approaches:

1. **Naive Approach**: Attempting to classify all 16 different poses with a single model
2. **Smart Approach**: Using specialized models for different pose categories:
   - Ball poses (3 classes)
   - FullBody poses (10 classes)
   - HalfBody poses (2 classes)

## Model Types

Several model architectures were explored:

1. **Neural Networks with K-Fold Cross-Validation**
2. **Neural Networks with Hyperparameter Tuning**
3. **K-Nearest Neighbors Classifiers**
4. **Multi-Layer Perceptron (MLP) Classifiers**

## Performance Summary

| Model Type | Approach | Category | Accuracy |
|------------|---------|----------|----------|
| Neural Network K-Fold | Naive | All Classes | 68.48% |
| Neural Network Tuned | Naive | All Classes | 64.06% |
| Neural Network K-Fold | Smart | Ball | 65.33% |
| Neural Network Tuned | Smart | Ball | 67.81% |
| Neural Network K-Fold | Smart | FullBody | 71.00% |
| Neural Network Tuned | Smart | FullBody | 65.00% |
| Neural Network K-Fold | Smart | HalfBody | 100.00% |
| Neural Network Tuned | Smart | HalfBody | 80.00% |
| K-Nearest Neighbors | Naive | All Classes | 69.40% |
| K-Nearest Neighbors | Smart | Ball | 73.00% |
| K-Nearest Neighbors | Smart | FullBody | 62.50% |
| K-Nearest Neighbors | Smart | HalfBody | 100.00% |
| MLP Classifier | Naive | All Classes | 67.69% |
| MLP Classifier | Smart | Ball | 73.70% |
| MLP Classifier | Smart | FullBody | 77.00% |
| MLP Classifier | Smart | HalfBody | 100.00% |

## Key Findings

- The "smart" approach generally outperforms the naive approach
- MLP Classifiers provide the best overall performance across categories
- HalfBody poses are easiest to recognize with 100% accuracy in most models
- The Ball and FullBody categories present more classification challenges

## Usage

To use the models for prediction:

```python
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load MoveNet
model_name = "movenet_thunder"
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

# Load one of the trained models
model = joblib.load('smart_ball_mlp.joblib')

# Function to get keypoints from an image
def get_keypoints(image):
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    
    # Run MoveNet
    model_fn = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model_fn(input_image)
    keypoints = outputs['output_0'].numpy()
    
    # Reshape for classifier input
    keypoints = keypoints.squeeze()
    keypoints = keypoints.reshape(1, 51)  # 17 keypoints × 3 values
    
    return keypoints

# Make a prediction
image = load_your_image()  # Load your image here
keypoints = get_keypoints(image)
prediction = model.predict(keypoints)
print(f"Predicted class: {prediction[0]}")
```

George Blagden, Carl Kaziboni, John Nocum
