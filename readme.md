# Predicción Financiera

Esta aplicación web permite obtener predicciones de precios de acciones utilizando diferentes algoritmos de aprendizaje automático. Los usuarios pueden ingresar el código de una acción y obtener el precio actual, junto con una predicción del precio futuro basada en tres algoritmos: Regresión Lineal, Regresión Polinómica y Forest Regressor. Además, la aplicación sugiere si es recomendable comprar o vender una acción basada en la predicción.

## Características

- **Visualización de precios**: Muestra el precio actual de una acción en tiempo real.
- **Predicción de precios**: Predice el precio futuro usando tres algoritmos de aprendizaje automático:
  - Regresión Lineal
  - Regresión Polinómica
  - Forest Regressor
- **Sugerencia de compra/venta**: Ofrece una sugerencia basada en la comparación entre el precio actual y el precio predicho.
- **Gráfico interactivo**: Visualiza el precio histórico de la acción en un gráfico dinámico.
  
## Tecnologías utilizadas

- **Flask**: Framework web para el desarrollo de la aplicación.
- **yFinance**: Para obtener datos históricos y en tiempo real de las acciones.
- **Scikit-learn**: Para los modelos de predicción (Regresión Lineal, Regresión Polinómica, y Random Forest Regressor).
- **Chart.js**: Para la visualización de gráficos interactivos.
- **Bootstrap 5**: Para el diseño y estilo responsivo.
  
## Instalación

1. Clona este repositorio:

    ```bash
    git clone https://github.com/GastonPaez/flask_yfinance
    ```

2. Navega al directorio del proyecto:

    ```bash
    cd flask_yfinance

    ```

3. Crea un entorno virtual (opcional pero recomendado):

    ```bash
    python -m venv venv
    ```

4. Activa el entorno virtual:

    - En **Windows**:

        ```bash
        venv\Scripts\activate
        ```

    - En **Mac/Linux**:

        ```bash
        source venv/bin/activate
        ```

5. Instala las dependencias del proyecto:

    ```bash
    pip install -r requirements.txt
    ```

    Asegúrate de que el archivo `requirements.txt` esté presente en el proyecto con las siguientes dependencias:

    ```txt
    Flask==2.1.1
    yfinance==0.1.70
    scikit-learn==1.0.2
    pandas==1.3.3
    matplotlib==3.4.3
    chart.js==3.7.0
    ```

6. Ejecuta la aplicación:

    ```bash
    python run.py
    ```

7. Abre tu navegador y ve a `http://127.0.0.1:5000/`.

## Uso

1. Ingresa el **símbolo de la acción** en el formulario de búsqueda. Por ejemplo, puedes ingresar "YPF" para obtener información sobre Apple.
2. La aplicación mostrará el **precio actual** de la acción y el **precio predicho** usando los tres algoritmos disponibles.
3. A continuación, obtendrás una **sugerencia de compra/venta** basada en la comparación entre el precio actual y el precio predicho.
4. Un gráfico interactivo mostrará la evolución histórica de los precios de la acción.

## Estructura del proyecto

El proyecto está organizado de la siguiente manera:

