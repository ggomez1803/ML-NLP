import pandas as pd
from xgboost import XGBRegressor
import joblib

# Cargar datos
dataTraining = pd.read_csv('./datasets/dataTrain_carListings.zip')

# Pre procesar datos
# Pre procesar los datos
top10_estados = list(dataTraining['State'].value_counts().head(10).index)

#Agrupar los estados que no estan en el top 10 en un solo estado
dataTraining['State'] = dataTraining['State'].apply(lambda x: x if x in top10_estados else 'Otro')

# Quitar espacios de la columna State
dataTraining['State'] = dataTraining['State'].apply(lambda x: x.strip())

# Convertir variables categóricas
dataTraining['State'] = dataTraining['State'].astype('category')
dataTraining['Make'] = dataTraining['Make'].astype('category')
dataTraining['Model'] = dataTraining['Make'].astype('category')

# Crear variables dummies
dataTraining = pd.get_dummies(dataTraining, columns=['State', 'Make', 'Model'], drop_first=True)

# Definir parámetros
estimators = 220
learning_rate = 0.1
max_depth = 20
weight = 34

# Definir X y
X_total = dataTraining.drop(['Price'], axis=1)
y_total = dataTraining['Price']

# Entrenar modelo
xgb = XGBRegressor(n_estimators=estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=weight)
xgb.fit(X_total, y_total)

# Guardar modelo
joblib.dump(xgb, 'xgb.pkl', compress=3)
print('Terminé')