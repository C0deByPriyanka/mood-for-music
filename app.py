from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from io import BytesIO

app = Flask(__name__)

# Load models
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

# Function to load and preprocess image from request
def load_and_preprocess_image(file, target_size=(48, 48)):
    img = image.load_img(BytesIO(file.read()), target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to recommend music based on mood
def recommend_music(mood_label):
    mapped_mood = mood_map.get(mood_label)
    mood_indices = [i for i, mood in enumerate(music_data['mood']) if mood == mapped_mood]
    if len(mood_indices) == 0:
        return []  # Return empty list if no music available for this mood
    
    recommended_indices = nn_model.kneighbors(X_music_scaled[mood_indices], return_distance=False)
    recommendations = music_data.iloc[recommended_indices[0]]
    return recommendations[['name', 'artist', 'album']].to_dict(orient='records')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is sent
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    # Load and preprocess the image directly from the request
    img = request.files['image']
    processed_image = load_and_preprocess_image(img)

    # Predict the mood
    prediction = mood_model.predict(processed_image)
    predicted_mood = mood_labels[np.argmax(prediction)]
    
    # Get recommendations
    recommendations = recommend_music(predicted_mood)
    
    # Return the result as JSON
    return jsonify({
        "predictedMood": predicted_mood,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
