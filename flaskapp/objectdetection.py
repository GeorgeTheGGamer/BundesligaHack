def detect_objects(image_path, class_id=37, model_name='mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'):
    """
    Detects objects in an image using TensorFlow Object Detection API.
    
    Args:
        image_path (str): Path to the image to detect objects in
        class_id (int, optional): COCO class ID to detect. Defaults to 37 (sports ball).
        model_name (str, optional): Model to use for detection. 
                                   Defaults to 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'.
    
    Returns:
        float: Detection score for the specified class ID (0 if not detected)
    """
    import subprocess
    import os
    import pathlib
    import sys
    import glob
    
    # Check if TensorFlow is installed, if not, install it
    try:
        import tensorflow as tf
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--pre", "tensorflow==2.*"])
        import tensorflow as tf
    
    # Check if pycocotools is installed, if not, install it
    try:
        import pycocotools
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools"])
    
    # Try to downgrade protobuf to version 3.20.0 which is known to work better with TensorFlow Object Detection API
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.0"])
    except subprocess.CalledProcessError:
        print("Warning: Could not install protobuf 3.20.0. Using existing version.")
    
    # Check and set up directory structure
    current_dir = pathlib.Path.cwd()
    if "models" in current_dir.parts:
        while "models" in current_dir.parts:
            os.chdir('..')
    elif not pathlib.Path('models').exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/tensorflow/models"])
    
    # Change directory to models/research
    research_path = os.path.join('models', 'research')
    os.chdir(research_path)
    
    # Add research directory to Python path
    research_path_abs = os.path.abspath(os.getcwd())
    if research_path_abs not in sys.path:
        sys.path.append(research_path_abs)
        print(f"Added {research_path_abs} to Python path")
    
    # Compile protobuf files
    proto_files = glob.glob("object_detection/protos/*.proto")
    for proto_file in proto_files:
        print(f"Compiling {proto_file}...")
        subprocess.check_call([
            "protoc",
            f"--python_out=.",
            proto_file
        ])
    
    # Verify proto compilation
    generated_files = glob.glob("object_detection/protos/*_pb2.py")
    print(f"Generated {len(generated_files)} protobuf files")
    
    # Install the object_detection package - skip if setup.py doesn't exist
    try:
        if os.path.exists("setup.py"):
            print("Installing object_detection package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
        else:
            print("No setup.py found, skipping installation of object_detection package.")
            # Instead, make sure the directory is in the Python path
            if os.path.abspath(".") not in sys.path:
                sys.path.insert(0, os.path.abspath("."))
    except subprocess.CalledProcessError:
        print("Warning: Could not install object_detection package. Using existing installation or Python path instead.")
        if os.path.abspath(".") not in sys.path:
            sys.path.insert(0, os.path.abspath("."))
    
    # Install tf_slim
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_slim"])
    
    # Force creation of missing __init__.py files to ensure proper imports
    for dir_path in [
        os.path.join("object_detection", "protos"),
        os.path.join("object_detection", "utils")
    ]:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            print(f"Creating missing {init_file}")
            with open(init_file, 'w') as f:
                pass
    
    # Import libraries
    import numpy as np
    import six.moves.urllib as urllib
    import tarfile
    import tensorflow as tf
    import zipfile
    
    from collections import defaultdict
    from io import StringIO
    from matplotlib import pyplot as plt
    from PIL import Image
    
    # Try imports with more robust error handling
    try:
        from object_detection.utils import ops as utils_ops
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as vis_util
        print("Successfully imported object_detection modules")
    except ImportError as e:
        print(f"Error importing object_detection modules: {e}")
        print("Attempting alternative import method...")
        
        # Try direct import of the compiled protobuf file
        import sys
        import importlib.util
        
        # Try to import the protobuf file directly
        proto_path = os.path.join(research_path_abs, "object_detection/protos/string_int_label_map_pb2.py")
        if os.path.exists(proto_path):
            module_name = "object_detection.protos.string_int_label_map_pb2"
            spec = importlib.util.spec_from_file_location(module_name, proto_path)
            string_int_label_map_pb2 = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = string_int_label_map_pb2
            spec.loader.exec_module(string_int_label_map_pb2)
            print("Successfully imported string_int_label_map_pb2 directly")
        
        # Now try to import the utils again
        try:
            from object_detection.utils import ops as utils_ops
            from object_detection.utils import label_map_util
            from object_detection.utils import visualization_utils as vis_util
            print("Successfully imported object_detection modules after direct protobuf import")
        except ImportError as e2:
            print(f"Still having import issues: {e2}")
            print("Using manual module loading...")
            
            # Manual module loading as a last resort
            utils_path = os.path.join(research_path_abs, "object_detection/utils")
            
            # Load label_map_util
            spec = importlib.util.spec_from_file_location(
                "label_map_util", 
                os.path.join(utils_path, "label_map_util.py")
            )
            label_map_util = importlib.util.module_from_spec(spec)
            sys.modules["object_detection.utils.label_map_util"] = label_map_util
            try:
                spec.loader.exec_module(label_map_util)
                print("Manually loaded label_map_util")
            except Exception as e:
                print(f"Error loading label_map_util: {e}")
            
            # Load visualization_utils
            spec = importlib.util.spec_from_file_location(
                "visualization_utils", 
                os.path.join(utils_path, "visualization_utils.py")
            )
            vis_util = importlib.util.module_from_spec(spec)
            sys.modules["object_detection.utils.visualization_utils"] = vis_util
            try:
                spec.loader.exec_module(vis_util)
                print("Manually loaded visualization_utils")
            except Exception as e:
                print(f"Error loading visualization_utils: {e}")
            
            # Load ops
            spec = importlib.util.spec_from_file_location(
                "ops", 
                os.path.join(utils_path, "ops.py")
            )
            utils_ops = importlib.util.module_from_spec(spec)
            sys.modules["object_detection.utils.ops"] = utils_ops
            try:
                spec.loader.exec_module(utils_ops)
                print("Manually loaded ops")
            except Exception as e:
                print(f"Error loading ops: {e}")
    
    # Add a description of the COCO classes for reference
    COCO_CLASSES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
        59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
        64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
        73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
        78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
        84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
    }
    
    def load_model(model_name):
        # Updated URL for TF2 models
        base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
        model_date = '20200711'  # This is the release date for TF2 models
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=model_name,
            origin=base_url + model_date + '/' + model_file,
            untar=True)
        
        # Search for the saved_model directory
        model_dir_path = pathlib.Path(model_dir)
        
        # Print the directory contents to help debug
        print(f"Downloaded to: {model_dir}")
        print("Directory contents:")
        for path in model_dir_path.glob("**/*"):
            print(path)
        
        # Look for all saved_model directories
        saved_model_paths = list(model_dir_path.glob("**/saved_model"))
        
        if not saved_model_paths:
            raise ValueError(f"Could not find saved_model directory in {model_dir}")
        
        # Use the first saved_model directory found
        saved_model_path = saved_model_paths[0]
        print(f"Found saved_model at: {saved_model_path}")
        
        model = tf.saved_model.load(str(saved_model_path))
        return model.signatures['serving_default']
    
    def run_inference_for_single_image(model, image):
        # Convert image to tensor and expand dimensions
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run inference
        output_dict = model(input_tensor)
        
        # Process the model outputs
        num_detections = int(output_dict.pop('num_detections'))
        
        # Handle the outputs more robustly to accommodate different model formats
        output_dict = {
            key: value[0, :num_detections].numpy() if len(value.shape) > 1 
            else value[0].numpy()
            for key, value in output_dict.items()
        }
        
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
        if 'detection_masks' in output_dict:
            try:
                # Try the newer API version first
                detection_masks_reframed = tf.image.crop_and_resize(
                    tf.expand_dims(output_dict['detection_masks'], axis=0),
                    output_dict['detection_boxes'],
                    tf.range(tf.shape(output_dict['detection_boxes'])[0]),
                    [image.shape[0], image.shape[1]]
                )
                detection_masks_reframed = tf.squeeze(detection_masks_reframed, axis=0)
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
                output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            except Exception as e:
                print(f"Error with newer API approach: {e}")
                try:
                    # Fall back to the older API approach
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        output_dict['detection_masks'], output_dict['detection_boxes'],
                        image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
                    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
                except Exception as e2:
                    print(f"Error with older API approach: {e2}")
                    print("Skipping mask reframing due to API compatibility issues")
        
        return output_dict
    
    def show_inference_score(model, image_path, class_id):
        try:
            image_np = np.array(Image.open(image_path))
            output_dict = run_inference_for_single_image(model, image_np)
            boxes = []
            classes = []
            scores = []
            for i, x in enumerate(output_dict['detection_classes']):
                if x == class_id and output_dict['detection_scores'][i] > 0.5:
                    classes.append(x)
                    boxes.append(output_dict['detection_boxes'][i])
                    scores.append(output_dict['detection_scores'][i])
            boxes = np.array(boxes)
            classes = np.array(classes)
            scores = np.array(scores)
            
            if len(scores) == 0:
                return 0
            
            return scores[0]
        except Exception as e:
            print(f"Error in show_inference_score: {e}")
            return 0
    
    # Define the path to label map
    PATH_TO_LABELS = os.path.join(research_path_abs, 'object_detection/data/mscoco_label_map.pbtxt')
    # Fallback if the original path doesn't exist
    if not os.path.exists(PATH_TO_LABELS):
        PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
        if not os.path.exists(PATH_TO_LABELS):
            print(f"Warning: Could not find label map at {PATH_TO_LABELS}. Downloading...")
            import requests
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
            r = requests.get(url)
            os.makedirs(os.path.dirname(PATH_TO_LABELS), exist_ok=True)
            with open(PATH_TO_LABELS, 'wb') as f:
                f.write(r.content)
    
    try:
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        print("Successfully created category index from label map")
    except Exception as e:
        print(f"Error loading label map: {e}")
        # Fallback to using COCO_CLASSES directly
        category_index = {id: {'id': id, 'name': name} for id, name in COCO_CLASSES.items()}
        print("Using fallback COCO_CLASSES as category index")
    
    try:
        print("Loading model...")
        detection_model = load_model(model_name)
        print("Model loaded successfully")
    
        # Check if the image exists
        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            result = show_inference_score(detection_model, image_path, class_id)
            print(f"Detection score for class {class_id} ({COCO_CLASSES.get(class_id, 'unknown')}): {result}")
            return result
        else:
            print(f"Warning: Image not found at {image_path}")
            return 0
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 0


