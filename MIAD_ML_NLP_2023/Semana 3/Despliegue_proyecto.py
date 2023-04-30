#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
# Leer archivo dataTrain_carListings
#dataTraining = pd.read_csv('./datasets/dataTrain_carListings.zip')

# Pre procesar datos
# Pre procesar los datos
top10_estados = list(dataTraining['State'].value_counts().head(10).index)

#Agrupar los estados que no estan en el top 10 en un solo estado
dataTraining['State'] = dataTraining['State'].apply(lambda x: x if x in top10_estados else 'Other')

# Quitar espacios de la columna State
dataTraining['State'] = dataTraining['State'].apply(lambda x: x.strip())

# Convertir variables categóricas
dataTraining['State'] = dataTraining['State'].astype('category')
dataTraining['Make'] = dataTraining['Make'].astype('category')
dataTraining['Model'] = dataTraining['Make'].astype('category')

# Crear variables dummies
dataTraining = pd.get_dummies(dataTraining, columns=['State', 'Make', 'Model'], drop_first=True)

# Definir X
X_total = dataTraining.drop(['Price'], axis=1)

def predict_price(year, mileage, state, make, model):

    reg = joblib.load(os.path.dirname(__file__) + '/xgb.pkl') 

    # Crear el dataframe para introducir al modelo de predicción
    x_test = pd.DataFrame(columns=X_total.columns)
    x_test.loc[0,'Year'] = year
    x_test.loc[0,'Mileage'] = mileage
    # Si no es uno de los estados más comunes del modelo, asigna "Other"
    if state in top10_estados:
        x_test.loc[0,'State_'+state] = 1
    else:
        x_test.loc[0,'State_Other'] = 1
    x_test.loc[0,'Make_'+make] = 1
    x_test.loc[0,'Model_'+model] = 1
    # Llenar el resto de columnas con 0
    x_test.fillna(0, inplace=True)
    # Eliminar columnas que no estén en el modelo original
    for col in x_test.columns:
        if col not in X_total.columns:
            x_test.drop(col, axis=1, inplace=True)

    # Predecir
    y_pred = reg.predict(x_test)

    return y_pred


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingrese los valores del vehículo')
        
    else:
            
            year = sys.argv[1]
            mileage = sys.argv[2]
            state = sys.argv[3]
            make = sys.argv[4]
            model = sys.argv[5]
    
            price = predict_price(year, mileage, state, make, model)
            
            print('Price: ', price)
        
