#Import libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tensorflow as tf
# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#Disease status
def trainingDiseaseStatus(training_path, testing_path):
    #Loading datasets
    df80 = pd.read_csv(training_path)
    df80 = df80.drop(columns=['Individual', 'Dataset', 'Disease Stage', 'Stage', 'Histology']) #these columns aren't needed for prediction. Disease stage, stage, and histology are outputs of the model, and not inputs so they are dropped when doing predictions.
    df20 = pd.read_csv(testing_path)
    df20 = df20.drop(columns=['Individual', 'Dataset', 'Disease Stage', 'Stage', 'Histology'])
    #Encoding the independent variables
    #Encoding smoking status as ordinal with ranked ordering as those that are smoking have a more likely chance to be at risk of lung cancer
    ordinal_order = ['Non-smoking', 'Ex-smoker', 'Current']
    oe = OrdinalEncoder(categories=[ordinal_order])
    ct = ColumnTransformer(transformers=[('encoder', OrdinalEncoder(categories=[ordinal_order]), [2])],
                           remainder='passthrough')
    df80 = np.array(ct.fit_transform(df80))
    df20 = np.array(ct.fit_transform(df20))
    #OneHotEncoding gender so that neither gender is weighted more than the other
    ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    df80 = np.array(ct2.fit_transform(df80))
    df20 = np.array(ct2.fit_transform(df20))
    X_train = df80[:, :-1]
    X_test = df20[:, :-1]
    y_train = df80[:, -1]
    y_test = df20[:, -1]
    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #Encoding the dependent variable as 0 for no cancer and 1 for cancer
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    #Random forest model
    classifier = RandomForestClassifier(n_estimators=150)
    #Training the model
    classifier.fit(X_train, y_train)
    #Prediction
    y_pred = classifier.predict(X_test)
    #Evaluating the model using AUC and accuracy
    auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
    acc = accuracy_score(y_test, y_pred)
    print("AUC: ", auc)
    print("ACC: ", acc)
    #Plotting the AUC curve 
    sklearn.metrics.plot_roc_curve(classifier, X_test, y_test) 
    plt.title('Disease Status Model ROC AUC', fontsize = 15)
    plt.show()
 trainingDiseaseStatus('80patientdata.csv', '20patientdata.csv')
#Early vs. Late 
def trainingEarlyvLate(training_path, testing_path): 
    #Loading datasets
    df80 = pd.read_csv(training_path)
    df80 = df80.drop(columns=['Individual', 'Dataset', 'Disease Status', 'Disease Stage', 'Histology'])
    df20 = pd.read_csv(testing_path)
    df20 = df20.drop(columns=['Individual', 'Dataset', 'Disease Status', 'Disease Stage', 'Histology'])
    #Encoding independent variables
    #Ordinal encoding for smoking (explanation in previous code cell)
    ordinal_order = ['Non-smoking', 'Ex-smoker', 'Current']
    oe = OrdinalEncoder(categories = [ordinal_order])
    ct = ColumnTransformer(transformers = [('encoder', OrdinalEncoder(categories=[ordinal_order]), [2])], remainder = 'passthrough')
    df80 = np.array(ct.fit_transform(df80))
    df20 = np.array(ct.fit_transform(df20))
    #OneHotEncoding gender 
    ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
    df80 = np.array(ct2.fit_transform(df80))
    df20 = np.array(ct2.fit_transform(df20))
    #Defining training and testing data
    X_train1 = df80[:, :-1]
    X_test1 = df20[:, :-1]
    y_train1 = df80[:, -1]
    y_test1 = df20[:,-1]
    #Label encoding dependent variable (cancer or no cancer) as 1 and 0
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train1 = le.fit_transform(y_train1)
    y_test1 = le.fit_transform(y_test1)
    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train1 = sc.fit_transform(X_train1)
    X_test1 = sc.transform(X_test1)
    #Random forest model
    classifier = RandomForestClassifier(n_estimators=150)
    #Training the model 
    classifier.fit(X_train1, y_train1)
    #Prediction
    y_pred1 = classifier.predict(X_test1)
    #Evaluating the model using AUC and accuracy
    auc = roc_auc_score(y_test1, classifier.predict_proba(X_test1)[:, 1])
    acc = accuracy_score(y_test1, y_pred1)
    print("AUC: ", auc)
    print("ACC: ", acc)
    #Plotting the AUC curve
    sklearn.metrics.plot_roc_curve(classifier, X_test1, y_test1) 
    plt.title('Early v Late Model ROC AUC', fontsize = 15)
    plt.show()
trainingEarlyvLate('80patientdatadiseasedonly.csv','20patientdatadiseasedonly.csv')
#Disease Stages I-IV
def trainingDiseaseStage(training_path, testing_path):
    #Loading datasets
    df80 = pd.read_csv(training_path)
    df80 = df80.drop(columns=['Individual', 'Dataset', 'Disease Status', 'Stage', 'Histology'])
    df20 = pd.read_csv(testing_path)
    df20 = df20.drop(columns=['Individual', 'Dataset', 'Disease Status', 'Stage', 'Histology'])
    #Encoding independent variables
    #Ordinal encoding smoking status
    ordinal_order = ['Non-smoking', 'Ex-smoker', 'Current']
    oe = OrdinalEncoder(categories=[ordinal_order])
    ct = ColumnTransformer(transformers=[('encoder', OrdinalEncoder(categories=[ordinal_order]), [2])],
                           remainder='passthrough')
    df80 = np.array(ct.fit_transform(df80))
    df20 = np.array(ct.fit_transform(df20))
    #One Hot Encoding gender 
    ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    df80 = np.array(ct2.fit_transform(df80))
    df20 = np.array(ct2.fit_transform(df20))
    X_train = df80[:, :-1]
    X_test = df20[:, :-1]
    #Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #Encoding the dependent variable
    y_train = df80[:, -1]
    y_train = y_train.astype('int')
    y_test = df20[:, -1]
    y_test = y_test.astype('int')
    #Random forest model
    classifier = RandomForestClassifier(n_estimators=1000)
    #Training the model 
    classifier.fit(X_train, y_train)
    #Prediction
    y_pred = classifier.predict(X_test)
    y_pred = y_pred.astype('int')
    #Evaluating the model using accuracy 
    acc = accuracy_score(y_test, y_pred)
    print("ACC: ", acc)
trainingDiseaseStage('80patientdatadiseasedonly.csv', '20patientdatadiseasedonly.csv')