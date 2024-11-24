import numpy as np
import pandas as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data=pd.read_csv("name.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
model=keras.Sequential()
model.add(keras.layers.Dense(6,activation="relu",input_dim=11))
model.add(keras.layers.Dense(6,activation="relu",input_dim=11))
