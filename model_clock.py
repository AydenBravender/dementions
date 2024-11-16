import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Parameters
model_path = "image_classifier.h5"  # Path to your saved model
img_size = (224, 224)  # Image size used during training
classes = [f"class{i}" for i in range(1, 7)]  # Adjust according to your dataset

# Load the model
model = load_model(model_path)

# Function to preprocess and classify an image
def classify_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = classes[predicted_class_idx]

    return predicted_class, predictions[0][predicted_class_idx]

# Example usage
if __name__ == "__main__":
    # Provide the path to an image you want to classify
    test_image_path = "path_to_your_test_image.jpg"

    try:
        predicted_class, confidence = classify_image(test_image_path)
        predicted_class = (100/7) * predicted_class
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
    except FileNotFoundError:
        print("Image file not found. Please provide a valid file path.")
