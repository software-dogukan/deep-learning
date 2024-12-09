import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# LabelEncoder ve OneHotEncoder tanımları
le = LabelEncoder()
ohe = OneHotEncoder()
sc = StandardScaler()

# Veri yükleme ve işleme
data = pd.read_csv("Cars_Dataset.csv")
data = data.iloc[:599, :]  # İlk 599 veriyi al

# Özellikleri ayırma
cars_name = data.iloc[:, 0].values
colors = data.iloc[:, 3].values
year_and_km = data.iloc[:, 1:3].applymap(lambda x: float(str(x).replace(",", ""))).values
types = data.iloc[:, 4].values
fuel = data.iloc[:, 5].values
target = data.iloc[:, -1].values  # Son sütun hedef değişken

# Kategorik verileri kodlama
cars_name_encoded = ohe.fit_transform(cars_name.reshape(-1, 1)).toarray()
colors_encoded = ohe.fit_transform(colors.reshape(-1, 1)).toarray()
types_encoded = ohe.fit_transform(types.reshape(-1, 1)).toarray()
fuel_encoded = ohe.fit_transform(fuel.reshape(-1, 1)).toarray()

# Özellikleri birleştirme
combined_data = np.hstack([cars_name_encoded, colors_encoded, types_encoded, fuel_encoded, year_and_km])

# Yeni DataFrame oluşturma
columns = (
    [f"car_name_{i}" for i in range(cars_name_encoded.shape[1])] +
    [f"color_{i}" for i in range(colors_encoded.shape[1])] +
    [f"type_{i}" for i in range(types_encoded.shape[1])] +
    [f"fuel_{i}" for i in range(fuel_encoded.shape[1])] +
    ["year", "kilometers"]
)
final_df = pd.DataFrame(combined_data, columns=columns)

# Hedef değişkeni 0 ve 1'e dönüştürme
y = le.fit_transform(target)

# Eğitim ve test verisini ayırma
x_train, x_test, y_train, y_test = train_test_split(final_df.values, y, test_size=0.33, random_state=0)

# Veriyi standartlaştırma
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Model oluşturma
model = keras.models.Sequential()
model.add(keras.layers.Dense(6, kernel_initializer="uniform", activation="relu", input_dim=X_train.shape[1]))
model.add(keras.layers.Dense(3, kernel_initializer="uniform", activation="relu"))
model.add(keras.layers.Dense(1, kernel_initializer="uniform", activation="sigmoid"))

# Modeli derleme
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Modeli test etme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
