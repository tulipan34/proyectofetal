import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM  # Necesita instalación: pip install fcmeans
import numpy as np
import os

# Crear carpeta "models" si no existe
os.makedirs("models", exist_ok=True)

# Cargar dataset
df = pd.read_excel("FGR_dataset.xlsx")
X = df[['C' + str(i) for i in range(1, 31)]]
y = df['C31']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Regresión Logística
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
with open("models/logistic_model.pkl", "wb") as f:
    pickle.dump(logistic_model, f)

# 2. Red Neuronal (MLPClassifier)
ann_model = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=1000)
ann_model.fit(X_train, y_train)
with open("models/ann_model.pkl", "wb") as f:
    pickle.dump(ann_model, f)

# 3. SVM
svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train, y_train)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# 4. Mapa Cognitivo Difuso (usamos Fuzzy C-Means como aproximación)
fcm = FCM(n_clusters=2)
fcm.fit(X_scaled)
# Mapeamos los clusters al target (esto es una forma básica de uso)
fcm_model = {"centers": fcm.centers, "scaler": scaler}
with open("models/fcm_model.pkl", "wb") as f:
    pickle.dump(fcm_model, f)

print("Todos los modelos han sido entrenados y guardados correctamente.")
