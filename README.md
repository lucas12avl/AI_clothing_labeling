# AI Clothing Labeling

## Introduction
This project is developed as part of the Artificial Intelligence course at the Autonomous University of Barcelona. 
It aims to create an agent capable of automatic image labeling, allowing intelligent natural language searches in an online store that constantly updates its product listings.
All using AI algorithims like KNN and Kmeans 

## Objectives
- Implement an automatic labeling system that assigns two types of labels: color and shape of the product.
- Enable users to perform searches using direct language, such as "red shirt" or "black sandals".

## Simplifications
Due to the complexity of the task, we will simplify the process by:
- Labeling only 8 types of clothing.
- Using the 11 basic universal colors to label the predominant colors of each clothing piece.

## Technologies Used
- Python programming language.
- K-means algorithm for unsupervised color labeling.
- K-NN (k-nearest neighbors) algorithm for supervised shape labeling.
- Fashion Product Images Dataset from Kaggle for low-resolution images (60x80 pixels).

## Setup and Installation
1. Clone the repository to your local machine.
2. unzip `train.zip` and `test.zip` this will be the images to train the AI and later test. (make sure there isn't no subfolders after unzip)


## Usage
Execute the `my_labeling.py` on your python IDE and run

## Tests
To test the KNN and Kmeans algorithims, execute `TestCases_kmeans.py` and `TestCases_knn.py`.

## Authors
 - Javier Comes Tello
 - Cristina Soler Bigorra
 - Lucas Dalmau Garcés