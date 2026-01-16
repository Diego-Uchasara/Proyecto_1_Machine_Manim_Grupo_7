# Linear vs Polynomial Regression Visualization 📉📈

Este proyecto utiliza [Manim](https://www.manim.community/) (Mathematical Animation Engine) para visualizar la teoría y la ejecución práctica de la Regresión Lineal frente a la Regresión Polinomial.

El código genera una animación que explica las funciones de hipótesis, las funciones de costo (MSE) y el proceso de descenso de gradiente, seguido de una simulación en tiempo real del ajuste de curvas sobre un dataset ruidoso.

## Autores - Grupo 7

* **Uchasara Huarachi, Diego David**
* **Roque Castillo, Franco Nicolas**
* **Montalvo Anaya, Diego Andres**

## Características de la Animación

1.  **Introducción y Presentación**: Créditos del equipo.
2.  **Teoría de Regresión Lineal**:
    * Visualización de la hipótesis $h_\theta(x) = w_0 + w_1 x$.
    * Definición del MSE (Mean Squared Error).
    * Ecuaciones de actualización del Gradiente.
3.  **Simulación Lineal**:
    * Entrenamiento en vivo con Descenso de Gradiente.
    * Resultado: Underfitting (Alto Sesgo).
4.  **Teoría de Regresión Polinomial**:
    * Expansión de características (Feature Expansion).
    * Hipótesis para grado $d$.
5.  **Simulaciones Polinomiales**:
    * **Grado 3**: Buen ajuste (Balanceado).
    * **Grado 5**: Bajo error en entrenamiento (Alta capacidad).
