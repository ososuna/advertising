# %% [markdown]
# # Regresión lineal simple en Python

# %%
import pandas as pd

# %%
data = pd.read_csv('../datasets/Advertising.csv')

# %%
data.head()

# %% [markdown]
# Se intentará buscar una relación lineal entre el gasto de TV y las ventas.

# %% [markdown]
# ## Regresión lineal simple

# %% [markdown]
# ## Paquete statsmodel

# %%
import statsmodels.formula.api as smf

# %% [markdown]
# Se define ventas (la variable predictora) en función de únicamente la televisión.

# %% [markdown]
# fit() sirve para calcular la recta que mejor se ajusta utilizando la técnica de minimizar la suma de los cuadrados de los errores

# %%
# Se manda como argumento la fórmula que se quiere llevar a cabo y el nombre del dataset
lm = smf.ols(formula='Sales~TV', data = data).fit()

# %% [markdown]
# El intercept es la 'a' de nuestro modelo lineal y el parámetro que acompañaría a TV  

# %%
lm.params

# %% [markdown]
# El modelo lineal predictivo sería
# sales = 7.032594 + 0.047537 * TV 

# %%
lm.pvalues

# %% [markdown]
# Los p valores son demasiado pequeños, podemos garantizar de que el parámetro no es 0 ni para el corte con la ordenada en el origen ni para el parámetro que acompaña al dinero gastado en TV.

# %% [markdown]
# Hay otro indicador importante del modelo que es la eficacia del modelo presente en el factor R**2 (la suma de los cuadrados totales).

# %%
lm.rsquared

# %% [markdown]
# Existe una variante entre que es el valor de R**2 ajustado, que se supone que va un poco mejor. Tiende a ser una modificación de un pequeño factor en función del número de elementos que se están estudiando.

# %%
lm.rsquared_adj

# %% [markdown]
# Visión general del modelo

# %%
lm.summary()

# %%
sales_pred = lm.predict(pd.DataFrame(data['TV']))
sales_pred

# %% [markdown]
# Se calculó la predicción de ventas para cada una de las filas basándonos en la ecuación que utiliza únicamente los costos de TV.

# %% [markdown]
# Podríamos hacer un plot de esta predicción vs los costos reales y mirar si la línea de tendencia se ajusta o no al valor de la predicción.

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
data.plot(kind ='scatter', x='TV', y='Sales')
plt.plot(pd.DataFrame(data['TV']), sales_pred, c='red', linewidth=2)
plt.show()
