import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow_hub as hub
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from ultralytics import YOLO

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Load the dog breed classification model
model = tf.keras.models.load_model('models/dog_breed.h5', compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

# Load YOLO model for dog detection
yolo_model = YOLO('yolo11n.pt')  # You can change to a larger model if needed

# List of dog breeds (ensure this matches your model's output classes)
breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 
          'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 
          'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 
          'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 
          'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 
          'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 
          'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 
          'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 
          'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 
          'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 
          'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 
          'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 
          'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 
          'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 
          'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 
          'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 
          'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 
          'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 
          'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 
          'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 
          'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 
          'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 
          'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 
          'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 
          'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 
          'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to preprocess image for breed classification
def preprocess_for_breed(cropped_img):
    # Resize the cropped image for the model input size
    cropped_img = cropped_img.resize((224, 224))  # Resize to model input size
    img_array = np.array(cropped_img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to detect dogs using YOLO
def detect_dogs(img_path):
    results = yolo_model(img_path)
    dog_bboxes = []

    for result in results:
        # Check for dog instances in the detected objects
        class_names = result.names
        for box in result.boxes:
            cls = int(box.cls[0])
            if class_names[cls].lower() == 'dog':
                dog_bboxes.append(box.xyxy[0].cpu().numpy())  # Save bounding boxes (x1, y1, x2, y2)

    return dog_bboxes

# Function to crop dog from the image based on bounding box
def crop_dog_image(img_path, bbox):
    img = Image.open(img_path)
    left, top, right, bottom = bbox
    cropped_img = img.crop((left, top, right, bottom))  # Crop based on the bounding box
    return cropped_img

@app.route('/', methods=['GET', 'POST'])
def index():
    global breeds
    predictions = []
    local_breeds = []
    confidences = []
    message = None  # Variable to hold the message for low confidence

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file and allowed_file(file.filename):
            # Save the uploaded image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Detect dogs using YOLO
            dog_bboxes = detect_dogs(img_path)

            if len(dog_bboxes) == 0:
                message = (
                    "No dogs detected. Please upload an image with dogs."
                    "Please ensure the image clearly shows the dog(s) and try again."
                )
            else:
                for bbox in dog_bboxes:
                    # Crop the detected dog from the image
                    cropped_img = crop_dog_image(img_path, bbox)

                    # Preprocess the cropped dog image for breed classification
                    img_array = preprocess_for_breed(cropped_img)

                    # Predict the breed
                    prediction = model.predict(img_array)
                    predicted_class_idx = np.argmax(prediction)  # Get the index of the highest prediction
                    confidence = np.max(prediction)  # Confidence score
                    
                    # Debugging predictions
                    print(f"Prediction: {prediction}")
                    print(f"Predicted Class Index: {predicted_class_idx}")
                    print(f"Confidence: {confidence}")
                    print(f"Total breeds: {len(breeds)}")

                    # Check if the confidence is above threshold and map breed
                    if confidence >= 0.5:
                        breed = breeds[predicted_class_idx] if predicted_class_idx < len(breeds) else "Unknown breed"
                    else:
                        breed = "Unknown breed"
                        message = (
                            "The detected dog(s) could not be identified with high confidence. "
                            "Please try using a clearer or higher-quality image."
                        )

                    # Append results for each dog
                    predictions.append(prediction)
                    local_breeds.append(breed)
                    confidences.append(confidence)

    # Zip breeds and confidences here
    breed_confidence_pairs = zip(local_breeds, confidences)

    return render_template('index.html', predictions=predictions, breed_confidence_pairs=breed_confidence_pairs, message=message)

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=False)
