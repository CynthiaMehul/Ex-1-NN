<H3>Cynthia Mehul J</H3>
<H3>212223240020</H3>
<H3>EX. NO.1</H3>
<H3>07.03.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv("C:/Users/admin/Downloads/Churn_Modelling.csv")
print(df)

# Finding Missing Values
print(df.isnull().sum())

# Check for Duplicates
df.duplicated()

# Detect Outliers
print(df['EstimatedSalary'].describe())

# Normalize the dataset
scaler=MinMaxScaler()
df_numeric = df.drop(columns=['Surname', 'Geography', 'Gender'])
df1=pd.DataFrame(scaler.fit_transform(df_numeric))
print(df1)

# Split the dataset into input and output
x = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# Split the dataset for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Print the training data and testing data
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:

Reading the dataset: 

![image](https://github.com/user-attachments/assets/5235939e-6a86-4b85-8724-f99558f5c56e)

Finding Missing Values:

![image](https://github.com/user-attachments/assets/07681ce8-e8ac-4615-88de-e4c84b3225c4)

Check for Duplicates:

![image](https://github.com/user-attachments/assets/3d153199-e7cc-4456-bfb3-6a7ba51c24fb)

Detect Outliers:

![image](https://github.com/user-attachments/assets/38a61889-df07-41d7-9e7c-2bcbf4d427f3)

Normalize the dataset:

![image](https://github.com/user-attachments/assets/e6482619-7ce1-4429-ad34-23ef0cbd9731)

Print the training data and testing data:

![image](https://github.com/user-attachments/assets/a86cc323-8312-408f-873a-e99ecaa50706)

![image](https://github.com/user-attachments/assets/d44900f6-6e1a-4014-ba4f-71a95313f582)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


