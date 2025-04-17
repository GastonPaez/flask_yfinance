from flask import Blueprint, render_template, request
from app.services.prediction import (
    predict_with_regression,
    predict_with_polynomial_regression,
    predict_with_forest_regressor
)
from app.utils.data import fetch_real_time_data, process_stock_data

main = Blueprint('main', __name__)  # Cambiar 'routes' a 'main'

@main.route('/', methods=["GET", "POST"])
def index():
    prediction_lineal = None
    prediction_polynomial = None
    prediction_forest = None
    current_price = None
    timestamps = []
    closing_prices = []
    algorithm = "Regresión Lineal"  # Algoritmo predeterminado
    stock_symbol = None

    if request.method == "POST":
        stock_symbol = request.form['stock_symbol'].upper()  

        # Obtiene los datos del mercado
        response = fetch_real_time_data(stock_symbol)

        if response is not None:
            print("Comienza el procesamiento de datos")
            timestamps, closing_prices = process_stock_data(response)
            if closing_prices:
                current_price = closing_prices[-1]
                # Calcula los algoritmos de predicción
                prediction_lineal = predict_with_regression(timestamps, closing_prices)
                prediction_polynomial = predict_with_polynomial_regression(timestamps, closing_prices)
                prediction_forest = predict_with_forest_regressor(timestamps, closing_prices)

    return render_template(
        'index.html',
        symbol=stock_symbol,
        current_price=current_price,
        prediction_lineal=prediction_lineal,
        prediction_polynomial=prediction_polynomial,
        prediction_forest=prediction_forest,
        timestamps=timestamps,
        closing_prices=closing_prices,
        algorithm=algorithm
    )
