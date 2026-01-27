import os
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "best_model.pkl")
threshold_path = os.path.join(BASE_DIR, "threshold.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(threshold_path, "rb") as f:
    threshold = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Variables simples
    Pclass = int(request.form['Pclass'])
    Sex_male = int(request.form['Sex_male'])
    Embarked_S = int(request.form['Embarked_S'])

    # Tranche d'âge
    age_tranche = request.form['TrancheAge']
    ta_13_18 = int(age_tranche == "[13,18]")
    ta_19_35 = int(age_tranche == "[19,35]")
    ta_36_60 = int(age_tranche == "[36,60]")
    ta_61_80 = int(age_tranche == "[61,80]")

    # Transformation Pclass en dummies
    Pclass_2 = int(Pclass == 2)
    Pclass_3 = int(Pclass == 3)

    # Construire le vecteur final (ordre exact du modèle)
    final_features = np.array([[
        Sex_male,
        ta_13_18,
        ta_19_35,
        ta_36_60,
        ta_61_80,
        Embarked_S,
        Pclass_2,
        Pclass_3
    ]])

    # Prédiction
    proba = model.predict_proba(final_features)[0, 1]
    prediction = int(proba >= threshold)
    label = "Survivant" if prediction == 1 else "Non survivant"

    return render_template(
        'index.html',
        prediction_text=f"{label} (Probabilité = {proba:.2f})"
    )


if __name__ == "__main__":
    app.run(debug=True)
