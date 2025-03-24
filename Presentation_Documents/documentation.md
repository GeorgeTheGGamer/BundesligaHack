# PRFAQ: Player Pose Classification System for Bundesliga Media Day Images
## DFL AI Hackathon Challenge 1

### Press Release

**DFL Implements AI-Powered System to Automate Player Media Day Image Classification**

*Frankfurt, Germany - March 2025*

The Deutsche Fußball Liga (DFL) today announced the implementation of a new AI-powered system that automatically classifies player media day photographs into specific pose categories. Developed during the Bundesliga AI Hackathon, this new system addresses a critical workflow challenge by eliminating the need for manual image sorting, which previously required significant time and resources.

"Our media day photoshoots generate thousands of images that need to be sorted into specific pose categories for our content management system," said [DFL Media Representative]. "This new AI solution can classify images with high accuracy, reducing what used to take days of manual work into a process that takes just minutes."

The innovative system uses advanced pose estimation technology from Google's MoveNet combined with specialized machine learning classifiers. By focusing on the player's skeletal structure rather than superficial image features, the system achieves robust classification regardless of variations in player appearance, jersey designs, or background conditions.

Early testing shows classification accuracy reaching 100% for certain pose categories and strong overall performance across all 16 specified pose classes. The DFL plans to fully integrate this technology into their media workflow for the upcoming season.

### Frequently Asked Questions

**Q: What problem does this system solve?**

A: The DFL captures thousands of player media day images that need to be sorted into 16 specific pose categories. This sorting was previously done manually, a time-consuming process that could take multiple days. Our system automates this classification, reducing processing time to minutes while maintaining high accuracy.

**Q: How does the system work?**

A: The system uses a two-stage approach:
1. First, Google's MoveNet Thunder pose estimation model extracts 17 skeletal keypoints from player images
2. Then, specialized machine learning classifiers analyze these keypoints to determine which of the 16 pose categories the image belongs to

This approach focuses on the fundamental pose structure rather than superficial image details, enabling better generalization to new images.

**Q: What level of accuracy does the system achieve?**

A: Our system achieves varied accuracy depending on the pose category:
- 100% accuracy for HalfBody pose classifications
- 77% accuracy for FullBody poses
- 73.7% accuracy for Ball-related poses
- Overall accuracy significantly outperforms conventional CNN approaches (48%)

The specialized "smart" approach using separate models for different pose types consistently outperforms a single generic classifier approach.

**Q: What machine learning models were evaluated?**

A: We evaluated several model architectures:
1. Neural Networks with K-Fold Cross-Validation
2. Neural Networks with Hyperparameter Tuning
3. K-Nearest Neighbors Classifiers
4. Multi-Layer Perceptron (MLP) Classifiers
5. Conventional Convolutional Neural Networks

The MLP Classifiers provided the best overall balance of performance, simplicity and consistency across all pose categories.

**Q: What are the 16 pose categories the system can classify?**

A: The system classifies images into the following categories:
- full_body
- half_body
- head_shot
- celebration
- arms_behind
- holding_ball
- holding_ball_45_degree_right
- holding_ball_45_degree_left
- crossed_arms_frontal
- crossed_arms_45_degree_right
- crossed_arms_90_degree_right
- crossed_arms_90_degree_left
- hands_on_hips_45_degree_right
- hands_on_hips_45_degree_left
- hands_on_hips_90_degree_right
- hands_on_hips_90_degree_left

**Q: What technical approach yielded the best results?**

A: Our "smart" approach that breaks down the classification problem into specialized models yielded the best results:
1. A Ball detection model for ball-related poses (3 classes)
2. A FullBody model for full-body poses (10 classes)
3. A HalfBody model for half-body poses (2 classes)

This specialization allows each model to focus on the most relevant features for its category, resulting in significantly improved performance compared to a single model approach.

**Q: What were some approaches that didn't work as well?**

A: Several approaches yielded valuable insights despite not being chosen for the final solution:
1. **Direct CNN Classification**: Traditional CNN models operating directly on the images achieved only 48% accuracy, likely because they focused too much on superficial features rather than pose structure.
2. **Hyperparameter-Tuned Neural Networks**: Surprisingly, more complex neural networks with tuned hyperparameters sometimes performed worse than simpler architectures, suggesting that the problem doesn't require highly complex models.
3. **Data Augmentation**: While theoretically beneficial, traditional image augmentation techniques (rotation, flipping) sometimes distorted the very pose aspects we were trying to classify.

**Q: How does the system handle edge cases?**

A: The system addresses several common edge cases:
1. **Unusual Poses**: By focusing on keypoints rather than raw images, the system can generalize to poses that are variations on the trained categories.
2. **Player Appearance Variations**: The keypoint-based approach is invariant to player characteristics like height, build, or skin tone.
3. **Jersey and Background Variations**: Unlike CNN approaches that might learn jersey patterns or backgrounds, our system focuses only on pose structure.
4. **Partial Occlusions**: The keypoint confidence scores help the model account for partially visible or occluded body parts.

**Q: How can the system be deployed in the DFL's workflow?**

A: The system can be deployed as:
1. A standalone application for batch processing of media day images
2. An API integrated into the DFL's content management system
3. A component in an automated image processing pipeline

The modular design allows for flexible deployment options based on the DFL's specific infrastructure needs.

**Q: What future improvements are planned?**

A: Several enhancements are being considered:
1. Expanding the training dataset with more diverse poses
2. Implementing ensemble methods to combine multiple classifiers
3. Adding temporal consistency checks for physically impossible poses
4. Creating a user-friendly interface for DFL staff to review and correct classifications when needed
5. Extending the system to classify video content for additional workflow automation

## Technical Implementation Details

### Data and Preprocessing

Our dataset consisted of 160 player images across 16 pose classes (10 images per class). All images were resized to be compatible with MoveNet's input requirements (256×256 pixels).

### Model Architecture

Our final solution uses a two-stage pipeline:

1. **Pose Extraction**:
   - MoveNet Thunder extracts 17 keypoints from each image
   - Each keypoint contains x, y coordinates and a confidence score
   - These 51 values (17 keypoints × 3 values) become our feature representation

2. **Classification**:
   - "Smart" approach with specialized models:
     - Ball model: MLP Classifier (3 classes, 73.7% accuracy)
     - FullBody model: MLP Classifier (10 classes, 77% accuracy)
     - HalfBody model: MLP Classifier (2 classes, 100% accuracy)

### MLP Classifier Configuration

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',             # ReLU activation function
    solver='adam',                 # Adam optimizer
    alpha=0.0001,                  # L2 regularization parameter
    batch_size='auto',             # Automatically determine batch size
    learning_rate='adaptive',      # Adaptive learning rate
    max_iter=1000,                 # Maximum number of iterations
    random_state=42                # For reproducibility
)
```

### Performance Analysis

| Model Type | Approach | Category | Accuracy |
|------------|---------|----------|----------|
| Neural Network K-Fold | Naive | All Classes | 68.48% |
| Neural Network Tuned | Naive | All Classes | 64.06% |
| K-Nearest Neighbors | Naive | All Classes | 69.40% |
| MLP Classifier | Naive | All Classes | 67.69% |
| Conventional CNN | Naive | All Classes | 48.00% |
| Neural Network K-Fold | Smart | Ball | 65.33% |
| Neural Network Tuned | Smart | Ball | 67.81% |
| K-Nearest Neighbors | Smart | Ball | 73.00% |
| MLP Classifier | Smart | Ball | 73.70% |
| Neural Network K-Fold | Smart | FullBody | 71.00% |
| Neural Network Tuned | Smart | FullBody | 65.00% |
| K-Nearest Neighbors | Smart | FullBody | 62.50% |
| MLP Classifier | Smart | FullBody | 77.00% |
| Neural Network K-Fold | Smart | HalfBody | 100.00% |
| Neural Network Tuned | Smart | HalfBody | 80.00% |
| K-Nearest Neighbors | Smart | HalfBody | 100.00% |
| MLP Classifier | Smart | HalfBody | 100.00% |

### Key Findings

1. **Keypoint-Based Representation**: The 17 keypoints from MoveNet provide a rich, informative representation for pose classification that generalizes well to new images.

2. **Specialized Models**: Breaking the problem into specialized models for different pose categories significantly improves performance compared to a single generic classifier.

3. **Model Complexity**: More complex models did not always yield better results. The MLP Classifier provided the best balance of complexity and performance.

4. **Performance by Category**: HalfBody poses were easiest to classify (100% accuracy), followed by FullBody poses (77%) and Ball-related poses (73.7%).

5. **Comparison to CNN**: The keypoint-based approach significantly outperformed conventional CNN classification (48% accuracy), suggesting that skeletal structure is more informative for pose classification than raw image features.

### Conclusion

Our pose classification system provides a robust solution for automating the categorization of player media day images. By focusing on skeletal structure rather than superficial image features, the system achieves high accuracy across all pose categories and generalizes well to new images. The specialized "smart" approach with separate models for different pose types consistently outperforms a single generic classifier approach.

This system can be immediately deployed to assist with media day image classification, significantly reducing the time and resources required for manual sorting.
