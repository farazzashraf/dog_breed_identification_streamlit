import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import os

# Register TensorFlow Hub layer as a custom object
tf.keras.utils.get_custom_objects()['KerasLayer'] = hub.KerasLayer
from tensorflow.keras.losses import CategoricalCrossentropy

# Register CategoricalCrossentropy explicitly if needed
tf.keras.utils.get_custom_objects()['CategoricalCrossentropy'] = CategoricalCrossentropy

# Streamlit Page Configuration
st.set_page_config(page_title="Dog Breed Classification", page_icon="🐕", layout="wide")

# Title and Description
st.title("Dog Breed Classification App 🐶")
st.markdown("""Upload an image of a dog to predict its breed using deep learning.
    The app will display the predicted breed name and confidence score.""")

# Sidebar for description
st.sidebar.header("How it Works:")
st.sidebar.markdown("""
    - **Step 1:** Upload an image of a dog.
    - **Step 2:** The model will predict the breed and display the confidence score.
    - **Step 3:** View the predicted breed name and its confidence level.
    """)

breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']

# Function to load the model
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'KerasLayer': hub.KerasLayer, 'CategoricalCrossentropy': CategoricalCrossentropy})
            print("Model loaded successfully!")
            return model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the trained model
model_path = 'models/dog_breed.h5'
model = load_model(model_path)

# Check if the model loaded successfully
if model is None:
    st.error("Failed to load the model. Please check the model path and try again.")
    st.stop()

# Upload image
uploaded_image = st.file_uploader("Choose a dog image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Load and display the image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", width=200)  # Display the image with full container width

    # Preprocess the image
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the breed
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)  # Get the index of the highest prediction
    confidence = np.max(prediction)  # Confidence score

    # Get the breed name
    predicted_breed = breeds[predicted_class_idx]  # Map class index to breed name

    # Display the results
    st.subheader("Prediction Result")
    
    # HTML and CSS styling for the breed name
    styled_breed_name = f"""
        <p style="font-family: 'Arial', sans-serif; font-size: 28px; font-weight: bold; color: #4CAF50;">
            {predicted_breed.capitalize()}
        </p>
    """
    
    # Display the styled breed name
    st.markdown(styled_breed_name, unsafe_allow_html=True)
    
    # Display confidence score with color styling
    st.markdown(f"<h4 style='color: #4CAF50;'>Confidence: {confidence:.2f}</h4>", unsafe_allow_html=True)

    # Confidence level text style
    if confidence > 0.8:
        st.markdown('<p style="color:green; font-size:20px;">High Confidence! 🎉</p>', unsafe_allow_html=True)
    elif confidence > 0.5:
        st.markdown('<p style="color:orange; font-size:20px;">Medium Confidence 🔶</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:red; font-size:20px;">Low Confidence ❗</p>', unsafe_allow_html=True)

else:
    st.info("Please upload an image of a dog to get the breed prediction.")
