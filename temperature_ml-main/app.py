from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    celsius = float(data["celsius"])

    prediction = model.predict([[celsius]])

    return jsonify({
        "celsius": celsius,
        "fahrenheit": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)