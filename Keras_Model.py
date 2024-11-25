import numpy as np
import pandas as pd
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
model=keras.models.Sequential()
model.add(keras.layers.Dense(6,init="uniform",activation="relu",input_dim=11))
model.add(keras.layers.Dense(3,init="uniform",activation="relu"))
model.add(keras.layers.Dense(1,init="uniform",activation="sigmoid"))
opt=keras.optimizers.Adam(learning_rate=0.01)
loss_fn=keras.losses.SparseCategoricalCrossentropy()
accuracy = keras.metrics.CategoricalAccuracy()
model.compile(loss=loss_fn,optimizer=opt,metrics=accuracy)
