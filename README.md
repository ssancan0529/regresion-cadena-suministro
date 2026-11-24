# Regresión aplicada a la cadena de suministro

Proyecto académico colaborativo de Optimización Numérica (ESPOL).

## Objetivo
Estimar parámetros de modelos de regresión lineal múltiple y no lineal para estudiar el precio de productos cosméticos a partir de costos de fabricación y envío, volumen de producción y tiempo de fabricación.

## Metodología
Estandarización de variables, minimización del error cuadrático, gradiente y Hessiano, verificación mediante Autograd, método de Newton, búsqueda de sección áurea, variante regularizada denominada Trust Region en el código y descenso del gradiente. Los resultados del modelo lineal se almacenan en SQL Server.

## Ejecución
1. Instalar Python, SQL Server Express y ODBC Driver 18.
2. Ejecutar `pip install -r requirements.txt`.
3. Preparar PROJECT_OPT_NUM con las tablas supply_chain_regression y resultados_modelo y cargar el conjunto de datos original. Revisar la conexión local.
4. Ejecutar `python FINAL_PROJECT_CODE.py`.

El conjunto de datos original y los resultados de ejecución no están adjuntos; no se afirman métricas predictivas ni resultados verificados.

## Script SQL
Crear primero PROJECT_OPT_NUM e importar el dataset original a dbo.supply_chain_model. Ejecutar `SQLQuery2.sql` en una base de pruebas: recrea la tabla de regresión y crea la tabla de resultados. No incluye el dataset original.
