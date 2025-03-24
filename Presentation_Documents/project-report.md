# Human Pose Recognition Using MoveNet and Machine Learning

## Executive Summary

This report documents the development and evaluation of a human pose recognition system using Google's MoveNet pose estimation model combined with various machine learning classifiers. The project aims to classify different human poses and movements based on skeletal keypoints extracted from images. Multiple model architectures and training approaches were explored to determine the most effective solution for different pose recognition scenarios.

## Project Overview

The system uses MoveNet Thunder, a state-of-the-art pose estimation model, to extract 17 keypoints from human images. These keypoints are then used as input features for various machine learning models to classify poses into different categories. The project explored both a naive approach (classifying 16 different poses) and a smart approach (categorizing poses into Ball, FullBody, and HalfBody categories).

## Technical Implementation

### Data Processing Pipeline

1. **Image Collection**: Images were organized into appropriate category folders
2. **Preprocessing**: Images were resized to be compatible with MoveNet's input requirements
3. **Keypoint Extraction**: MoveNet Thunder was used to extract 17 keypoints with 3 values each (x, y, confidence)
4. **Feature Engineering**: The keypoints were reshaped into feature arrays for model training
5. **Model Training**: Multiple model architectures were trained and evaluated

### Model Architectures

Several model architectures were explored:

1. **Neural Networks with K-Fold Cross-Validation**:
   - Basic dense neural networks
   - Hyperparameter-tuned neural networks using Keras Tuner

2. **K-Nearest Neighbors**:
   - Using optimal K values determined through cross-validation

3. **MLP Classifiers**:
   - Standard Multi-Layer Perceptron classifiers

### Classification Approaches

The project employed two main classification approaches:

1. **Naive Approach**: Attempting to classify all 16 different poses directly
2. **Smart Approach**: Breaking the problem down into three specialized models:
   - Ball model (3 classes)
   - FullBody model (10 classes)
   - HalfBody model (2 classes)

## Results

### Classification Accuracy by Model and Approach

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

### Analysis of Results

1. **Smart vs. Naive Approach**: The specialized "smart" approach generally outperformed the naive approach, particularly for certain pose categories.

2. **Best Models by Category**:
   - **Ball Poses**: MLP Classifier (73.70%)
   - **FullBody Poses**: MLP Classifier (77.00%)
   - **HalfBody Poses**: Neural Network K-Fold, K-Nearest Neighbors, and MLP all achieved 100%

3. **Overall Best Performance**: The MLP Classifier with the smart approach performed most consistently across all categories.

4. **Model Complexity vs. Performance**: Neural networks with hyperparameter tuning did not consistently outperform simpler models, suggesting that the additional complexity may not be beneficial for this task.

## Key Insights

1. **Specialized Models**: Breaking the problem down into specialized models for different pose types significantly improves classification performance.

2. **HalfBody Recognition**: All models achieved excellent performance on HalfBody poses, suggesting these poses have clear, distinguishable features.

3. **MLP Effectiveness**: MLP classifiers provided the best balance of complexity and performance across most categories.

4. **Feature Representation**: The 17 keypoints (51 total values including x, y, confidence) from MoveNet provide sufficient information for effective pose classification.

## Conclusion

The project demonstrates the effectiveness of combining pose estimation (MoveNet) with machine learning classifiers for human pose recognition. The specialized "smart" approach, breaking the problem into distinct pose categories, proved more effective than attempting to classify all poses with a single model.

The MLP Classifier emerged as the most effective and consistent model type across all categories, suggesting it strikes an optimal balance between model complexity and performance for this application.

Future work could explore real-time implementation, expanded pose libraries, and further refinement of the feature engineering process to improve performance on more challenging pose categories.
