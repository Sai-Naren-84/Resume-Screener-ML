# Resume Screening & Job Role Prediction System

## Overview

This project uses Machine Learning to predict a candidate's job role based on their skills.

The system:

* Reads resume skills from text input
* Converts text into numerical features using TF-IDF
* Trains a Logistic Regression model
* Predicts suitable job roles such as:

  * Software Engineer
  * Data Scientist
  * ML Engineer
  * Web Developer
  * Backend Developer

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression
* Pickle

## Project Structure

Resume-Screener-ML/

* dataset.csv
* train.py
* predict.py
* model.pkl
* vectorizer.pkl
* requirements.txt
* README.md

## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Predict a role:

python predict.py

## Future Improvements

* Larger resume dataset
* Resume-to-job matching score
* Web interface using Flask
* Better ML models
