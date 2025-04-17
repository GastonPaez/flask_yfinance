from flask import Blueprint, render_template, request
from app.services.prediction import (
    predict_with_regression,
    predict_with_polynomial_regression,
    predict_with_forest_regressor
)
from app.utils.data import fetch_real_time_data, fetch_monthly_data, process_stock_data

main = Blueprint('main', __name__)

@main.route('/', methods=["GET", "POST"])
def index():
    prediction_lineal = None
    prediction_polynomial = None
    prediction_forest = None
    current_price = None
    timestamps = []
    closing_prices = []
    timestamps_monthly = []
    closing_prices_monthly = []
    algorithm = "Regresión Lineal"
    stock_symbol = None

    if request.method == "POST":
        stock_symbol = request.form['stock_symbol'].upper()

        # Datos intradía
        intraday_data = fetch_real_time_data(stock_symbol)
        if intraday_data is not None:
            timestamps, closing_prices = process_stock_data(intraday_data)
            if closing_prices:
                current_price = closing_prices[-1]
                prediction_lineal = predict_with_regression(timestamps, closing_prices)
                prediction_polynomial = predict_with_polynomial_regression(timestamps, closing_prices)
                prediction_forest = predict_with_forest_regressor(timestamps, closing_prices)

        # Datos mensuales
        monthly_data = fetch_monthly_data(stock_symbol)
        if monthly_data is not None:
            timestamps_monthly = list(monthly_data.index.strftime("%Y-%m-%d"))
            closing_prices_monthly = list(monthly_data["Close"].dropna())

    return render_template(
        'index.html',
        symbol=stock_symbol,
        current_price=current_price,
        prediction_lineal=prediction_lineal,
        prediction_polynomial=prediction_polynomial,
        prediction_forest=prediction_forest,
        timestamps=timestamps,
        closing_prices=closing_prices,
        timestamps_monthly=timestamps_monthly,
        closing_prices_monthly=closing_prices_monthly,
        algorithm=algorithm
    )