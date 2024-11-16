import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

class ImageClassifier:
    def __init__(self, model_path, img_size=(224, 224), classes=None):
        self.model_path = model_path
        self.img_size = img_size
        self.classes = classes or [f"class{i}" for i in range(1, 7)]  # Default class names

        # Load the pre-trained model
        self.model = load_model(self.model_path)

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses an image for classification.
        """
        # Load the image and resize it to the target size
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img) / 255.0  # Normalize the pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def classify_image(self, image_path):
        """
        Classifies the image and returns the predicted class and confidence.
        """
        img_array = self.preprocess_image(image_path)

        # Make predictions using the model
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get index of max probability
        predicted_class = self.classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        return predicted_class, confidence

# # Example usage
# if __name__ == "__main__":
#     # Initialize the classifier with model path and class names
#     model_path = "/home/ayden/Desktop/aydenprj/nathacks/image_classifier.h5"
#     test_image_path = "/home/ayden/Desktop/aydenprj/nathacks/model_cnn/dataset/class6/274_0.png"

#     classifier = ImageClassifier(model_path)

#     try:
#         predicted_class, confidence = classifier.classify_image(test_image_path)
#         print(f"Predicted Class: {predicted_class}")
#         print(f"Confidence: {confidence:.2f}")
#     except FileNotFoundError:
#         print("Image file not found. Please provide a valid file path.")
