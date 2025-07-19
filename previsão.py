
# Biblioteca 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# DADOS EXEMPLO:
horas_estudo = np.array ([[1], [2], [3], [4], [5]])
notas = np.array ([2,4,2,4,2])

# MODELO DE REGRESÃO LINEAR
modelo = LinearRegression()

# TREINAR AJUSTAR O MODELO DE DADOS
modelo.fit(horas_estudo, notas)

# Novas horas para previsão
horas_novas = np.array([[6]])

# Faz a previsão usando o modelo, não o numpy
notas_previstas = modelo.predict(horas_novas)

print(notas_previstas)

print(f"Se estudar 6 horas, a nota prevista é: {notas_previstas[0]:.2f}")

# opcional: mostrar gráfico
plt.scatter(horas_estudo, notas, color='blue', label="Dados reais")
plt.plot(horas_estudo, modelo.predict(horas_estudo), color='red', label="Linha de predição")
plt.xlabel("Horas de estudo")
plt.ylabel("Nota")
plt.title("Relação entre horas de estudo e nota")
plt.legend()
plt.show()