from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('wine_model.joblib')

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction=f"Predicted Wine Quality: {round(prediction[0], 2)}")

if __name__ == '__main__':
    app.run(debug=True)
