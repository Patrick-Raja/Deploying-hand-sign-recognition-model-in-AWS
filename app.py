from flask import Flask, render_template, request
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/predict', methods=['POST'])
def predict():
    if 'pic' in request.files:
        uploaded_file = request.files['pic']
        uploaded_file_path = os.path.join('uploads', 'temp_image.jpg')
        uploaded_file.save(uploaded_file_path)
        prediction = process_image(uploaded_file_path)

        return render_template('index.html', result=str(prediction))
    else:
        return render_template('index.html', result="No file uploaded")

def process_image(file_path):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                   11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'R', 17: 'Q', 18: 'S', 19: 'T', 20: 'U',
                   21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    data_aux = []
    x_ = []
    y_ = []
    frame = cv2.imread(file_path)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        return str(predicted_character)
    else:
        return "No hands detected"

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
