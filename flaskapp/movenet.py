import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the model
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

def movenet(input_image):
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def load_image(image_path):
    """Loads and preprocesses an image for MoveNet.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        A preprocessed image tensor ready for model input.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_with_pad(image, input_size, input_size)
    return image

def detect_pose(image_path):
    """Loads an image, detects pose, and returns processed keypoints.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Processed keypoints array of shape [17, 3] containing y, x coordinates and confidence scores.
    """
    # Load and preprocess the image
    image = load_image(image_path)
    
    # Run pose detection
    keypoints = movenet(image)
    
    # Convert to numpy array and remove dimensions of size 1
    keypoints = np.array(keypoints)
    keypoints = np.squeeze(keypoints, axis=tuple(i for i, dim in enumerate(keypoints.shape) if dim == 1))
    
    return keypoints


