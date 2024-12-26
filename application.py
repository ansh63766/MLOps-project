from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info("GET request received. Rendering home page.")
        return render_template('home.html')
    else:
        logging.info("POST request received. Collecting input data from form.")
        
        # Collect form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        # Log the data before creating the DataFrame
        logging.info(f"Received Data - Gender: {data.gender}, Ethnicity: {data.race_ethnicity}, "
                     f"Parental Level of Education: {data.parental_level_of_education}, Lunch: {data.lunch}, "
                     f"Test Preparation Course: {data.test_preparation_course}, "
                     f"Reading Score: {data.reading_score}, Writing Score: {data.writing_score}")
        
        pred_df = data.get_data_as_data_frame()

        # Log the DataFrame created from the input data
        logging.info(f"Data as DataFrame: {pred_df}")

        logging.info("Before Prediction: Initializing PredictPipeline.")
        predict_pipeline = PredictPipeline()
        
        logging.info("Mid Prediction: Calling predict method.")
        try:
            results = predict_pipeline.predict(pred_df)
            logging.info("After Prediction")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('home.html', results="Error occurred during prediction")

        # Return the prediction results
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
