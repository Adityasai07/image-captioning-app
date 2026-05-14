from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

app = Flask(__name__)

# ---------------- LOAD MODELS ----------------
model = load_model("model_ep30_bs32.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

# ---------------- VGG16 MODEL ----------------
vgg = VGG16(weights='imagenet')
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

# ---------------- UTILS ----------------
def idx_to_word(index):
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return None


def extract_features(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return vgg.predict(image, verbose=0)


def predict_caption(feature):
    in_text = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')

        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    path = os.path.join("static/uploads", file.filename)
    file.save(path)

    feature = extract_features(path)
    caption = predict_caption(feature)

    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run(debug=True)