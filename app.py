from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('pothole_detection_model.h5')

# Define a function to make predictions
def predict_pothole(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    return result[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    prediction = predict_pothole(file_path)
    result = 'Normal' if prediction < 0.5 else 'Pothole'


    # Redirect to the result page with the prediction
    return redirect(url_for('result', prediction=result))

@app.route('/result/<prediction>')
def result(prediction):
    return render_template('results.html', prediction=prediction)

@app.route('/pothole.html')
def pothole_page():
    return render_template('pothole.html')

@app.route('/signup.html')
def signup_page():
    return render_template('signup.html')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create 'uploads' folder if not exists
    app.run(debug=True)