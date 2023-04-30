from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from Despliegue_proyecto import predict_price

app = Flask(__name__)
api = Api(app, version='1.0',
          title= 'Predictor de precios de autos',
          description= 'API para predecir el precio de carros usados usando un modelo XGBoost')

ns = api.namespace('predict', description='Predicción de precios de vehículos')

parser = api.parser()

parser.add_argument('year', type=int, required=True, help='Año del vehículo', location='args')
parser.add_argument('mileage', type=int, required=True, help='Kilometraje del vehículo', location='args')
parser.add_argument('state', type=str, required=True, help='Estado del vehículo', location='args')
parser.add_argument('make', type=str, required=True, help='Marca del vehículo', location='args')
parser.add_argument('model', type=str, required=True, help='Modelo del vehículo', location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class CarPricingApi(Resource):
    
        @api.doc(parser=parser)
        @api.marshal_with(resource_fields)
        def get(self):
            args = parser.parse_args()
    
            year = args['year']
            mileage = args['mileage']
            state = args['state']
            make = args['make']
            model = args['model']
    
            return {
            "result": predict_price(year, mileage, state, make, model)
            }, 200
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8080)
    
        