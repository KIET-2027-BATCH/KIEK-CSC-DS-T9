from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open("car_price_model.pkl", "rb"))  # Save your trained model as car_price_model.pkl

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        company = data["company"]
        model_name = data["model"]
        mileage = float(data["mileage"])
        year = int(data["year"])

        # Process input data
        input_data = np.array([[mileage, year]])  # Modify based on model input

        # Predict price
        predicted_price = model.predict(input_data)[0]

        return jsonify({"predicted_price": round(predicted_price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
