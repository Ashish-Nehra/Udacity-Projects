# Flower Image Classifier
This project is an image classifier that is designed to recognize different species of flowers. The classifier is trained on a dataset called the Oxford 102 Flower Dataset, which consists of 102 different flower categories. The trained model can be used in various applications, such as a mobile app that identifies flowers using the device's camera.

## Dataset
The code uses the Oxford 102 Flower Dataset, which can be downloaded from the following link: Oxford 102 Flower Dataset

## Installation and Dependencies
Before running the code, make sure you have the following dependencies installed:

TensorFlow
TensorFlow Datasets
TensorFlow Hub
You can install these dependencies using the following command:

```
pip install tensorflow tensorflow-datasets tensorflow_hub
```
Usage

Install the dependencies specified as mentioned above.

## Load the image dataset and create a pipeline: 
Used Oxford 102 Flower Dataset using TensorFlow Datasets. It then creates a pipeline for the dataset, which involves preprocessing the images and batching them for training, validation, and testing.

## Build and train the image classifier: 
Build a classifier using a pre-trained MobileNetV2 model from TensorFlow Hub. It freezes the pre-trained layers and adds additional layers on top for fine-tuning. The classifier is trained using the training set and evaluated on the validation set.

## Use the trained model for inference: 
The code provides a function for processing images and making predictions. You can use this function to perform inference on flower images.

## Additional Files
The code uses a JSON file called "label_map.json" to map class indices to class names. Make sure to have this file in the same directory as the code. It is used to display the predicted class names during inference.
