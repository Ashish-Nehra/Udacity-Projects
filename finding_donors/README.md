# Finding Donors for CharityML
This repository contains the code for a supervised learning project called "Finding Donors for CharityML". The goal of this project is to develop a model that can accurately predict whether an individual earns more than $50,000 per year, in order to identify potential donors for a charity organization.

## Project Structure
The project is organized into the following sections:

## Exploring the Data: 
This section involves loading the dataset, exploring its contents, and performing initial data exploration.

## Preparing the Data: 
In this section, the data is preprocessed and transformed to make it suitable for training machine learning models.

## Evaluating Model Perform* ance: 
Various supervised learning models are trained and evaluated using performance metrics such as accuracy and F-score.

## Improving Results: 
Model tuning techniques are applied to improve the performance of the best-performing model.

## Getting Started
To run the code in this project, follow these steps:

Clone the repository to your local machine or download the source code files.

Make sure you have the following dependencies installed:

* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn

## Code Overview
The code in the Jupyter Notebook is divided into different sections, each focusing on a specific aspect of the project. Here's a brief summary of each section:

## Exploring the Data: 
This section imports the necessary libraries, loads the dataset, and displays the first few records to get an idea of the data.

## Data Exploration: 
In this section, various statistics and visualizations are used to explore the dataset, such as the total number of records, income distribution, and percentage of individuals earning more than $50,000.

## Preparing the Data: 
The data is preprocessed and transformed to make it suitable for training machine learning models. Skewed features are log-transformed, and numerical features are normalized using MinMaxScaler. Categorical features are one-hot encoded.

## Data Preprocessing: 
This section performs one-hot encoding on the transformed data and encodes the target variable to numerical values.

## Shuffle and Split Data: 
The data is split into training and testing sets using the train_test_split function from scikit-learn.

## Evaluating Model Performance: 
Various supervised learning models, including Decision Tree, Support Vector Machine (SVM), and Random Forest, are trained and evaluated on different subsets of the data to assess their performance.

## Model Tuning: 
The Random Forest model is tuned using GridSearchCV to find the optimal hyperparameters that maximize the F-score.

## Feature Importance: 
The feature importances of the trained Random Forest model are extracted and visualized to identify the most significant features.

## Final Model Evaluation: 
The final model trained on the full dataset and the reduced dataset (with the top 5 most important features) is evaluated and compared using accuracy and F-score.

## Conclusion
The "Finding Donors for CharityML" project demonstrates the application of supervised learning algorithms to predict potential donors for a charity organization. By analyzing the provided dataset, preprocessing the data, training different models, and tuning the best-performing model, we are able to achieve a model with improved accuracy and F-score. The feature importance analysis provides insights into the key features that contribute to the prediction task.

Feel free to explore the code and experiment with different models and techniques to further enhance the performance of the model or adapt it for your own use case.