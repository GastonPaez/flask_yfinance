import os
from flask import Flask, render_template, request
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    prediction = None
    current_price = None
    predicted_price = None
    timestamps = []
    closing_prices = []

    if request.method == "POST":
        stock_symbol = request.form['stock_symbol']
        
        # Obtener datos históricos y procesarlos
        response = fetch_real_time_data(stock_symbol)    
        print("Information" in response)    
        print(type(response))
        if "Information" in response:
            error = "API rate limit per day"
            timestamps, closing_prices, current_price, predicted_price, prediction = error, error, 0, 0, "Waiting"
            print("error api")
        
        else:
            if response:
                print("entro al if")
                timestamps, closing_prices = process_stock_data(response)
                current_price = closing_prices[-1]  # Último precio de cierre
                predicted_price = predict_with_regression(timestamps, closing_prices)  # Predicción basada en regresión
                prediction = 'Buy' if predicted_price > current_price else 'Sell'

    return render_template(
        'index.html',
        prediction=prediction,
        current_price=current_price,
        predicted_price=predicted_price,
        timestamps=timestamps,
        closing_prices=closing_prices
    )
    
def fetch_real_time_data(stock_symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval=5min&apikey=RB4FK71JRVS10X15'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print('Error fetching data:', e)
        return None

def process_stock_data(data):
    try:
        time_series = data["Time Series (5min)"]
        timestamps = list(time_series.keys())[:10]  # Últimos 10 registros
        timestamps.reverse()  # Orden cronológico
        closing_prices = [float(time_series[time]["4. close"]) for time in timestamps]
        return timestamps, closing_prices
    except KeyError as e:
        print("Error processing data:", e)
        return [], []

def predict_with_regression(timestamps, closing_prices):
    """ El código toma los precios de cierre de las últimas 10 observaciones y los convierte en dos conjuntos:
    X: Representa el tiempo como índices numéricos. Por ejemplo, si tienes 10 datos, los índices serán [0, 1, 2, ... 9].
    y: Son los precios de cierre reales correspondientes a esos momentos de tiempo.
    
    Estos datos son los que el modelo utiliza para identificar patrones.
    
    Args:
        timestamps (datetime): fecha y hora
        closing_prices (float): precio de cierre

    Returns:
        float: prediccion
    """
    # Convertir timestamps en valores numéricos (ej. [0, 1, 2, ...])
    X = np.array(range(len(timestamps))).reshape(-1, 1)
    y = np.array(closing_prices)

    # Entrenar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Predecir el próximo precio (siguiente paso en el tiempo)
    next_time = np.array([[len(timestamps)]])  # Puntero al siguiente índice
    predicted_price = model.predict(next_time)
    return predicted_price[0]

if __name__ == '__main__':
    app.run(debug=True)
