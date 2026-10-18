**Introduction:**

The Diabetes Disease Prediction project is a machine learning application developed using Python to predict whether a person falls into a diabetes-positive or diabetes-negative outcome category based on health-related input features.

The project uses the Logistic Regression algorithm to train a classification model on a diabetes dataset. It evaluates the model using accuracy, 
a classification report, and a confusion matrix to understand its prediction performance.

**Features:**

1.Loads and processes the diabetes dataset using Pandas.

2.Splits the dataset into training and testing sets.

3.Trains a Logistic Regression classification model.

4.Predicts outcomes for test data.

5.Calculates model accuracy.

6.Generates a classification report.

7.Visualizes the confusion matrix using a heatmap.

8.Predicts an outcome for a custom sample input.

**Technologies Used:**

-Python – Main programming language.

-Pandas – Dataset loading and data handling.

-NumPy – Numerical operations and input preparation.

-Scikit-learn – Model training, prediction, and evaluation.

-Matplotlib – Plotting and visualization.

-Seaborn – Confusion matrix heatmap visualization.

**Installation and Setup:**

 1.Clone the Repository
 
   git clone https://github.com/Haamdinishabegum/disease_prediction

 2.Navigate to the project directory

   cd Disease-Prediction

3.Install Dependencies

Install the required Python libraries:

pip install -r requirements.txt

4.Run the Project

Execute the Python script:

python disease_prediction.py

**How It Works:**

1.Loads the diabetes dataset from diabetes.csv.

2.Separates the input features from the target column, Outcome.

3.Splits the data into 80% training data and 20% testing data.

4.Trains the Logistic Regression model using the training data.

5.Predicts outcomes for the test dataset.

6.Evaluates performance using accuracy and a classification report.

7.Displays a confusion matrix heatmap.

8.Predicts an outcome for a sample input and prints the result.
