import keras
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = keras.models.load_model('../image_classification_model')

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = tf.image.decode_image(file.read(), channels=3)
        img = tf.image.resize(img, [32, 32])
        img = img.numpy() / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class])
        object_name = class_names[predicted_class]

        return jsonify({
            'class': int(predicted_class),
            'confidence': confidence,
            'object_name': object_name
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
