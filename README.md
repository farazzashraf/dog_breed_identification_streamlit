# Dog Breed Identification App

This is a Flask web application that uses a deep learning model to classify dog breeds from uploaded images. The model uses YOLO for dog detection and a custom-trained TensorFlow model for breed classification.

## Requirements

Before running the app, make sure you have the following dependencies installed:

- Python 3.10
- pip (Python package installer)

## Setup Instructions

Follow these steps to set up and run the application locally or deploy it to a platform like Render.

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/dog_breed_identification.git
cd dog_breed_identification

python -m venv venv
# For Windows use `venv\Scripts\activate`
pip install -r requirements.txt

python app2.py
