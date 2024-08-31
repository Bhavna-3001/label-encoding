# label-encoding
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
data = pd.read_csv("C:/Users/bhavn/OneDrive/Desktop/datasets/breast_cancer.csv")
data.head()
data['diagnosis'].value_counts()
data.head()
label_encode = LabelEncoder()
labels=label_encode.fit_transform(data.diagnosis)
data.head()
data['target']=labels
data.head()
data['target'].value_counts()

# Label encoding is a technique used in machine learning and data analysis to convert categorical variables into numerical format. It is particularly useful when working with 
# algorithms that require numerical input, as most machine learning models can only operate on numerical data.
#data link -> https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset
