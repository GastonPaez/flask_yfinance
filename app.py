from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
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
                # Calcula los algoritmos de prediccion
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

#================================================#
#========== OBTENER DATOS DE YFINANCE ===========#
#================================================#

def fetch_real_time_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        history = stock.history(period="1d", interval="5m")  # Datos intradía
        if not history.empty:
            print(f"Datos obtenidos correctamente para {stock_symbol}")
            return history
    except Exception as e:
        print(f"Error obteniendo datos de {stock_symbol}: {str(e)}")
    return None

#================================================#
#============ PROCESAMIENTO DE DATOS ============#
#================================================#

def process_stock_data(data):
    try:
        timestamps = list(data.index.strftime("%Y-%m-%d %H:%M"))[-10:]  # Últimos 10 registros
        closing_prices = list(data["Close"].dropna())[-10:]  # Últimos 10 precios de cierre
        return timestamps, closing_prices
    except KeyError as e:
        print("Error procesando datos:", e)
        return [], []
    
#================================================#
#=========== ALGORITMOS DE PREDICCION ===========#
#================================================#

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
    X = np.array(range(len(timestamps))).reshape(-1, 1)
    y = np.array(closing_prices)
    model = LinearRegression()
    model.fit(X, y)
    next_time = np.array([[len(timestamps)]])
    predicted_price = model.predict(next_time)
    return predicted_price[0]

def predict_with_polynomial_regression(timestamps, closing_prices):
    """ El código aplica regresión polinómica para predecir futuros precios de cierre basándose en datos históricos.

    X: Representa el tiempo como índices numéricos y se transforma en un conjunto de características polinómicas.
    y: Son los precios de cierre reales utilizados para ajustar el modelo polinómico.

    Este método permite capturar relaciones no lineales en los datos, mejorando la precisión en escenarios donde los precios tienen variaciones más complejas.

    Args:
        timestamps (datetime): fecha y hora
        closing_prices (float): precios de cierre

    Returns:
        float: predicción del precio de cierre
    """ 
    X = np.array(range(len(timestamps))).reshape(-1, 1)
    y = np.array(closing_prices)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    next_time = np.array([[len(timestamps)]])
    next_time_poly = poly.transform(next_time)
    predicted_price = model.predict(next_time_poly)
    return predicted_price[0]

def predict_with_forest_regressor(timestamps, closing_prices):
    """ El código utiliza un modelo de bosque aleatorio (Random Forest Regressor) para predecir futuros precios de cierre basándose en datos históricos.

    X: Representa el tiempo como índices numéricos y puede incluir características adicionales derivadas de precios anteriores.
    y: Son los precios de cierre reales que sirven como referencia para el modelo de aprendizaje basado en múltiples árboles de decisión.

    Este enfoque de aprendizaje de conjunto mejora la capacidad de generalización del modelo, capturando patrones complejos en los datos.

    Args:
        timestamps (datetime): fecha y hora
        closing_prices (float): precios de cierre

    Returns:
        float: predicción del precio de cierre
"""
    X = np.array(range(len(timestamps))).reshape(-1, 1)
    y = np.array(closing_prices)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    next_time = np.array([[len(timestamps)]])
    predicted_price = model.predict(next_time)
    return predicted_price[0]

if __name__ == '__main__':
    app.run(debug=True)