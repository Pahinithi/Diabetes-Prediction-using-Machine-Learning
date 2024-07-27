# Diabetes Prediction using Machine Learning

This project aims to predict whether a person has diabetes based on various medical attributes using machine learning techniques. The project utilizes a Support Vector Machine (SVM) for classification, trained on the PIMA Diabetes dataset.

## Project Structure

The project consists of the following files:

- `index.html`: Frontend of the application where users can input their medical information and get predictions.
- `app.py`: Backend of the application using Flask to handle user inputs and make predictions using the trained model.
- `diabetes_prediction_model.pkl`: Serialized machine learning model used for making predictions.
- `diabetes.csv`: Dataset used for training the model.
- `diabetes-prediction-with-machine-learning.ipynb`: Jupyter Notebook containing the steps for data preprocessing, model training, and evaluation.

## Dataset

The dataset used in this project is the PIMA Diabetes dataset. It contains the following columns:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0: Non-Diabetic, 1: Diabetic)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pahinithi/Diabetes-Prediction-using-Machine-Learning.git
   cd Diabetes-Prediction-using-Machine-Learning
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open a web browser and go to `http://127.0.0.1:5000`.

## Usage

1. Enter the required medical information in the form on the web page.
2. Click the "Predict" button to get the prediction result.
3. The prediction will indicate whether the person is diabetic or not.

## Model Training

The model training process is documented in the Jupyter Notebook (`diabetes-prediction-with-machine-learning.ipynb`). The following steps are performed:

1. Data Collection and Analysis
   - Loading the dataset
   - Inspecting the dataset
   - Statistical analysis

2. Data Preprocessing
   - Handling missing values
   - Data standardization

3. Model Training
   - Splitting the data into training and test sets
   - Training an SVM classifier

4. Model Evaluation
   - Evaluating the model's accuracy on training and test data

5. Saving the Model
   - Saving the trained model using `pickle`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset is provided by the National Institute of Diabetes and Digestive and Kidney Diseases.
- The project is powered by Python, Flask, and various data science libraries such as NumPy, Pandas, and scikit-learn.

## Author

Pahirathan Nithilan (https://pahinithi.github.io/nithi99) 

##  Live Demo

The application is hosted on Hugging Face Spaces: Diabetes Prediction App (https://huggingface.co/spaces/Nithi99/diabetes_prediction_app)

