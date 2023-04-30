#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from Proyecto1_xgb import X_total, top10_estados

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
        
