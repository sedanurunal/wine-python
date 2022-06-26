from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
wine=load_wine()

#Conver to pandas dataframe
data=pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])

#Check data with info function
data.info()
from sklearn.preprocessing import StandardScaler

#Remove target columns.
x = data.loc[:,data.columns != 'target'].values
y = data.loc[:,['target']].values

#Scale the data
y=pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.29,random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
acc_score_test=accuracy_score(y_test,predictions)

print(acc_score_test)