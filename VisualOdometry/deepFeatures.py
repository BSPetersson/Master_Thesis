import tensorflow.keras.applications.vgg16 as VGG16
import tensorflow.keras.applications.mobilenet_v2 as MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

vgg_model = VGG16.VGG16(weights='imagenet', include_top=False)
mobilenet_v2_model = MobileNetV2.MobileNetV2(weights='imagenet', include_top=False)

def extract_keypoint_patches(image, num_scales=8, scale_factor=0.8333, patch_size=32, nfeatures=6000):
    if len(image.shape) != 2:
        raise ValueError("Input image should be a 2D grayscale numpy array")

    # Create the image pyramid
    pyramid = [image]
    for i in range(1, num_scales):
        scaled_image = cv2.resize(pyramid[-1], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        pyramid.append(scaled_image)

    # Find FAST key points for each scale of the image pyramid
    fast = cv2.FastFeatureDetector_create(threshold=20)
    keypoints = []
    for i, scaled_image in enumerate(pyramid):
        keypoints_scale = fast.detect(scaled_image, None)
        keypoints.extend([(kp, i) for kp in keypoints_scale])

    # Sort the keypoints based on response values and select the top nfeatures keypoints
    keypoints.sort(key=lambda x: x[0].response, reverse=True)
    keypoints = keypoints[:nfeatures]

    # Extract 32x32 image patches for each key point at the correct scale
    patches_coords = []
    for keypoint, scale_idx in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        size = int(patch_size / 2)

        # Scale the keypoint coordinates back to the original image
        x_unscaled, y_unscaled = int(x * (1 / scale_factor) ** scale_idx), int(y * (1 / scale_factor) ** scale_idx)

        if y_unscaled - size < 0 or x_unscaled - size < 0 or y_unscaled + size >= image.shape[0] or x_unscaled + size >= image.shape[1]:
            continue

        patch = image[y_unscaled - size:y_unscaled + size, x_unscaled - size:x_unscaled + size]
        patches_coords.append((patch, (x_unscaled, y_unscaled)))

    return patches_coords


def process_patch(patch, point, model_name):
    if len(patch.shape) == 2:  # Check if the image is grayscale
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)  # Convert grayscale image to RGB

    x = img_to_array(patch)
    if model_name == 'vgg':
        x = VGG16.preprocess_input(x)
    elif model_name == 'mobilenet_v2':
        x = MobileNetV2.preprocess_input(x)
    return x, point

def get_image_features(img, model_name='vgg', batch_size=3000, nfeatures=6000):
    if model_name not in ['vgg', 'mobilenet_v2']:
        raise ValueError("model_name should be either 'vgg' or 'mobilenet_v2'")

    des = []
    kp = []

    patches_coords = extract_keypoint_patches(img, num_scales=8, scale_factor=0.8333, patch_size=32, nfeatures=nfeatures)

    with ThreadPoolExecutor() as executor:
        processed_patches = list(executor.map(process_patch, [patch for patch, _ in patches_coords], [point for _, point in patches_coords], [model_name] * len(patches_coords)))

    batch_patches = []
    batch_points = []
    for i, (x, point) in enumerate(processed_patches):
        batch_patches.append(x)
        batch_points.append(point)

        if len(batch_patches) == batch_size:
            print("{}/{}".format(i, len(processed_patches)))
            if model_name == 'vgg':
                batch_features = vgg_model.predict(np.array(batch_patches))
            elif model_name == 'mobilenet_v2':
                batch_features = mobilenet_v2_model.predict(np.array(batch_patches))

            for i, features in enumerate(batch_features):
                x, y = batch_points[i]
                kp.append(cv2.KeyPoint(x, y, 1))

                # Normalize and scale the features
                features = features.flatten()
                features = (features - np.min(features)) / (np.max(features) - np.min(features))
                features = (features * 255).astype(np.uint8)
                des.append(features)

            print(point)

            batch_patches = []
            batch_points = []

    if batch_patches:
        if model_name == 'vgg':
            batch_features = vgg_model.predict(np.array(batch_patches))
        elif model_name == 'mobilenet_v2':
            batch_features = mobilenet_v2_model.predict(np.array(batch_patches))

        for i, features in enumerate(batch_features):
            x, y = batch_points[i]
            kp.append(cv2.KeyPoint(x, y, 1))

            # Normalize and scale the features
            features = features.flatten()
            features = (features - np.min(features)) / (np.max(features) - np.min(features))
            features = (features * 255).astype(np.uint8)
            des.append(features)

    des = np.array(des)
    kp = np.array(kp)

    return (kp, des)
