# scripts/music_recommendation.py

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib

# Mood recommendation
mood_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load mood classification model
mood_model = load_model('models/mood_cnn_model.h5')

# Load music recommender model
nn_model = joblib.load('models/music_recommender.pkl')

# Load music data
music_data = pd.read_csv('data/data_moods.csv')

# Feature selection
features = ['popularity', 'length', 'danceability', 'acousticness', 'energy', 
            'instrumentalness', 'liveness', 'valence', 'loudness', 
            'speechiness', 'tempo', 'key', 'time_signature']
X_music = music_data[features]
y_music = music_data['mood']

# Normalize features
scaler = StandardScaler()
X_music_scaled = scaler.fit_transform(X_music)

def recommend_music_from_image(image_path):
    processed_image = load_and_preprocess_image(image_path)
    # Check prediction
    prediction = mood_model.predict(processed_image)

    # Convert prediction to mood label
    predicted_mood = mood_labels[np.argmax(prediction)]

    # Get recommendations
    recommendations = recommend_music(predicted_mood)
    return { 'predicted_mood': predicted_mood, 'recommendations': recommendations[['name', 'artist', 'album']] }

# Function to recommend music based on mood
def recommend_music(mood_label):
    mood_map = {
        'happy': 'Happy',
        'sad': 'Sad',
        'fear': 'Sad',
        'surprise': 'Energetic',
        'angry': 'Energetic',
        'disgust': 'Energetic',
        'neutral': 'Calm'
    }
    
    mood_indices = [i for i, mood in enumerate(y_music) if mood == mood_map.get(mood_label)]
    if len(mood_indices) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no music available for this mood
    recommended_indices = nn_model.kneighbors(X_music_scaled[mood_indices], return_distance=False)
    return music_data.iloc[recommended_indices[0]]

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def main():
    image_path = 'data/test/angry/PrivateTest_88305.jpg'
    print(recommend_music_from_image(image_path))

main()


