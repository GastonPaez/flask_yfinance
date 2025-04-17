import yfinance as yf

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

def process_stock_data(data):
    try:
        timestamps = list(data.index.strftime("%Y-%m-%d %H:%M"))[-10:]  # Últimos 10 registros
        closing_prices = list(data["Close"].dropna())[-10:]  # Últimos 10 precios de cierre
        return timestamps, closing_prices
    except KeyError as e:
        print("Error procesando datos:", e)
        return [], []
