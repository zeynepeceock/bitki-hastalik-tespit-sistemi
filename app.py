from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("model/plant_disease_model.keras")

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def format_result(predicted_class):
    if "healthy" in predicted_class.lower():
        disease_status = "Sağlıklı"
    else:
        disease_status = "Hastalıklı"

    if predicted_class.startswith("Pepper"):
        plant_type = "Biber"
    elif predicted_class.startswith("Potato"):
        plant_type = "Patates"
    elif predicted_class.startswith("Tomato"):
        plant_type = "Domates"
    else:
        plant_type = "Bilinmeyen"

    disease_name = predicted_class.replace("___", " ")
    disease_name = disease_name.replace("__", " ")
    disease_name = disease_name.replace("_", " ")

    return plant_type, disease_status, disease_name

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    plant_type = None
    disease_status = None
    disease_name = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_path = filepath

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]

            plant_type, disease_status, disease_name = format_result(predicted_class)
            confidence = round(np.max(prediction) * 100, 2)

    return render_template(
        "index.html",
        image_path=image_path,
        plant_type=plant_type,
        disease_status=disease_status,
        disease_name=disease_name,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)