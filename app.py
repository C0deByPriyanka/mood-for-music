from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pandas as pd
import joblib

app = Flask(__name__)

# Load models and data
mood_model = load_model('models/mood_cnn_model.h5')
nn_model = joblib.load('models/music_recommender.pkl')

# Load music data
music_data = pd.read_csv('data/data_moods.csv')

# Feature selection
features = ['popularity', 'length', 'danceability', 'acousticness', 'energy', 
            'instrumentalness', 'liveness', 'valence', 'loudness', 
            'speechiness', 'tempo', 'key', 'time_signature']
X_music = music_data[features]

# Normalize features
scaler = StandardScaler()
X_music_scaled = scaler.fit_transform(X_music)

# Mood labels
mood_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

mood_map = {
        'happy': 'Happy',
        'sad': 'Sad',
        'fear': 'Sad',
        'surprise': 'Energetic',
        'angry': 'Energetic',
        'disgust': 'Energetic',
        'neutral': 'Calm'
    }

def recommend_music(mood_label):
    mood_indices = [i for i, mood in enumerate(music_data['mood']) if mood == mood_map.get(mood_label)]
    if len(mood_indices) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no music available for this mood
    recommended_indices = nn_model.kneighbors(X_music_scaled[mood_indices], return_distance=False)
    return music_data.iloc[recommended_indices[0]]

def load_and_preprocess_image(image_file, target_size=(48, 48)):
    img = Image.open(image_file)
    img = img.resize(target_size)
    img = np.array(img)
    if img.shape[-1] != 3:  # Ensure 3 channels
        img = np.stack([img] * 3, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

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

    if file:
        # Load and preprocess image directly from request
        processed_image = load_and_preprocess_image(file)
        prediction = mood_model.predict(processed_image)
        print(prediction)
        predicted_mood = mood_labels[np.argmax(prediction)]

        # Recommend music
        recommendations = recommend_music(predicted_mood)
        recommendations_list = recommendations[['name', 'artist', 'album']].to_dict(orient='records')

        return render_template('result.html', mood=predicted_mood, recommendations=recommendations_list)

if __name__ == '__main__':
    app.run(debug=True)
