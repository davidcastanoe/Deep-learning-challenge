# Deep Learning Challenge

## Background 
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization.

## Instructions
### Step 1: Preprocess the data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables.
7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train & Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

    - Dropping more or fewer columns.
    - Creating more bins for rare occurrences in columns.
    - Increasing or decreasing the number of values for each bin.
    - Add more neurons to a hidden layer.
    - Add more hidden layers.
    - Use different activation functions for the hidden layers.
    - Add or reduce the number of epochs to the training regimen.

## Overview of the analysis
The objective of this analysis is to leverage machine learning techniques, particularly neural networks, to develop a binary classifier that can predict the success of funding applicants for Alphabet Soup, a nonprofit foundation. By utilizing the features provided in the dataset, the goal is to create a robust model that can accurately classify whether an applicant will be successful if funded by Alphabet Soup. This predictive tool will assist Alphabet Soup's decision-making process in selecting applicants with the highest potential for success in their ventures, thereby optimizing the allocation of resources and maximizing the impact of the foundation's funding initiatives.


## Data Preprocessing:
**What variable(s) are the target(s) for your model?**

The target variable for the model is "IS_SUCCESSFUL." This variable indicates whether the money provided by Alphabet Soup was used effectively by the funding applicants. The goal of the model would be to predict whether future applicants would be successful in utilizing the funding.

**What variable(s) are the features for your model?**

The Features variables for the model are 

_APPLICATION_TYPE: Alphabet Soup application type
AFFILIATION: Affiliated sector of industry
CLASSIFICATION: Government organization classification
USE_CASE: Use case for funding
ORGANIZATION: Organization type
STATUS: Active status
INCOME_AMT: Income classification
SPECIAL_CONSIDERATIONS: Special considerations for application
ASK_AMT: Funding amount requested_

**What variable(s) should be removed from the input data because they are neither targets nor features?**

Variables that should be removed from the input data because they are neither targ  ets nor features include: **EIN and NAME:** These are identification columns and are not relevant for predicting the success of funding applicants.

## Compiling, Training, and Evaluating the Model:

**How many neurons, layers, and activation functions did you select for your neural network model, and why?**

In our initial model, we opted for a configuration consisting of two layers: the first with 80 neurons utilizing the Rectified Linear Unit (ReLU) activation function, and the second with 30 neurons also using ReLU activation. Additionally, we included an output layer with the Sigmoid activation function. We chose ReLU for both hidden layers due to its effectiveness in capturing nonlinear relationships within the data, while Sigmoid is utilized in the output layer for its suitability in binary classification tasks. These settings represent the foundational features of our initial model. By employing these predetermined features, we aim to gain insight into the behavior of the model and understand its performance based on these patterns.

<img src="Images\layers.png" style="width:800px">

**Were you able to achieve the target model performance?**

Yes, we successfully achieved our target model performance. Below are the results of our model accuracy optimization.

<img src="Images\results.png" style="width:800px">

**What steps did you take in your attempts to increase model performance?**

The steps taken included adjusting the cutoff classification metrics from 800 to 300, thereby introducing an additional classification category, C7000-777. Instead of dropping the table name, we implemented a cutoff lower than 100, effectively reducing the number of unique values from 19,568 to 31. This adjustment enables the model to handle a smaller dataset, enhancing its accuracy. 

<img src="Images\number columns.png" style="height:400px">
<img src="Images\Name.png" style="width:800px">

## Summary

The deep learning model aimed to predict whether funding applicants for Alphabet Soup, a nonprofit foundation, would be successful. It used various details about the applicants to make these predictions. While the model performed okay, there's room for improvement. One suggestion is to try a different approach, like using a gradient boosting algorithm such as XGBoost or LightGBM. These methods can handle the data more effectively and might give better results. Gradient boosting algorithms are particularly well-suited for tabular data like ours, as they can handle categorical variables and interactions between features efficiently. They often provide high accuracy and are relatively easy to interpret, making them a promising choice for improving the model's performance. By trying these methods, we hope to make more accurate predictions, helping Alphabet Soup choose successful applicants more reliably.

## Credits
The research support to make this challenge successful comes from Chat Gpt, Google, Stackoverflow.

## Contact
If there are any questions or concerns, I can be reached at:

> [Github](https://github.com/Davidcastanoe)

> [LinkedIn](https://www.linkedin.com/in/davidcastanoe/)