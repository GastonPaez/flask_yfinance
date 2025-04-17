import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

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