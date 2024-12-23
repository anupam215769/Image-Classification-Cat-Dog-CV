import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Import necessary libraries for Random Forest model
import joblib
from skimage import color, transform, feature

# Set the title of the app
st.title('Image Classification: Cat vs Dog')

# Add an option to select the model
model_option = st.selectbox('Choose a model', ('CNN Model (Tiny VGG)', 'Random Forest Model'))

# Load the trained model (use caching to prevent reloading on every run)
@st.cache_resource
def load_trained_model(model_option):
    if model_option == 'CNN Model (Tiny VGG)':
        model = tf.keras.models.load_model('model_1.keras')
    else:
        # For Random Forest, load the model saved as a joblib file
        model = joblib.load('random_forest.joblib')
    return model

model = load_trained_model(model_option)

# Create a file uploader component
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.')

    if model_option == 'CNN Model (Tiny VGG)':
        # Preprocess the image before prediction for CNN
        def preprocess_image_cnn(image):
            # Resize the image to match model's expected sizing
            size = (128, 128)  # Your model input size
            image = image.resize(size)
            img_array = tf.keras.utils.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescaling as done during training
            return img_array

        # Preprocess the image
        img_array = preprocess_image_cnn(image)

        # Make a prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0]

        # Interpret the prediction
        if confidence >= 0.5:
            prediction_label = 'Dog'
            probability = confidence
        else:
            prediction_label = 'Cat'
            probability = 1 - confidence

    else:
        # Preprocess the image before prediction for Random Forest
        def preprocess_image_rf(image):
            # Convert to grayscale
            image = image.convert('L')
            # Resize the image
            image = image.resize((128, 128))
            # Convert to numpy array
            image_array = np.array(image)
            # Normalize pixels between 0 and 1
            image_array = image_array / 255.0
            # Extract HOG features
            hog_features = feature.hog(
                image_array,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True,
            )
            return hog_features.reshape(1, -1)  # Reshape for prediction

        # Preprocess the image
        img_features = preprocess_image_rf(image)

        # Make a prediction
        prediction = model.predict(img_features)
        confidence = model.predict_proba(img_features)

        # The Random Forest Classifier uses labels encoded as integers
        # Map back from integers to 'Cat' or 'Dog'
        predicted_class = prediction[0]  # 0 or 1
        probability = confidence[0][predicted_class]  # Probability of the predicted class

        if predicted_class == 1:
            prediction_label = 'Dog'
        else:
            prediction_label = 'Cat'

    confidence_percentage = probability * 100

    with col2:
        # Display the prediction result
        st.write(f"### Prediction: {prediction_label}")

        if confidence_percentage > 80:
            st.success(f"High confidence ({confidence_percentage:.2f}%)")
        elif 40 <= confidence_percentage <= 80:
            st.warning(f"Moderate confidence ({confidence_percentage:.2f}%)")
        else:
            st.error(f"Low confidence ({confidence_percentage:.2f}%)")