# Import necessary modules
from flask import Flask, request, render_template
import pickle
import numpy as np

# Create a Flask application instance
app = Flask(__name__, template_folder='.')

# Load the trained model
with open("diabetes_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html") # Render the index.html template

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    skin_thickness = float(request.form["skin_thickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    diabetes_pedigree_function = float(request.form["diabetes_pedigree_function"])
    age = float(request.form["age"])

    # Combine inputs into a single list
    input_data = np.asarray([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Determine the output message
    if prediction[0] == 0:
        prediction_text = 'The person is not diabetic'
    else:
        prediction_text = 'The person is diabetic'

    # Pass the prediction value to the template
    return render_template("index.html", prediction=prediction_text)

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True) # Run the application in debug mode for development
