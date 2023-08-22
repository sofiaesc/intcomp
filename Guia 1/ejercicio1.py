import random
import numpy as np
import csv
import matplotlib.pyplot as plt

# ------------------------------------- #
# ----------- ENTRENAMIENTO ----------- #
# ------------------------------------- #
trn = np.loadtxt('./data/OR_trn.csv',delimiter=',')

# ------------ COEFICIENTES ----------- #

yd = [] # Salida esperadas
for i in range(len(trn)): 
    fila = trn[i]
    cant_e = len(fila) - 1 # Cantidad de entradas
    yd.append(fila[cant_e]) # Salidas esperadas
    aux = [-1]
    for j in range(cant_e):
        aux.append(fila[j])
    trn[i] = aux # Vector de entradas

w = [] # Vector de pesos
for i in range(cant_e+1):
    aux = random.uniform(-0.5,0.5)
    w.append(aux)

# ------------- ALGORITMO ------------- #
epoca = 0
epoca_max = 10 # Máximo de iteraciones
cont_error = 0
gamma = 0.1 # Velocidad de aprendizaje
errores = np.zeros(epoca_max)
perc_error_max = 0.02 # Porcentaje máximo de error
error_max = perc_error_max * len(trn) # Cantidad máxima de errores

while (epoca < epoca_max):
    cont_error = 0
    for patron in range(len(trn)):
        x = trn[patron] # Entradas
        y = w @ x
        y = np.sign(y)

        error = yd[patron] - y
        if error != 0: cont_error += 1 

        w = w + gamma*error*x # Corrijo pesos

    errores[epoca] = cont_error
    if cont_error < error_max: break
    epoca += 1

# Gráfica de los errores por epoca:
fig,ax = plt.subplots()
plt.plot(range(epoca_max),errores)
ax.set(xlabel = 'Epoca',ylabel='Cantidad de errores')
plt.title('Cantidad de errores en entrenamiento')
ax.grid()
plt.show()

# ------------------------------------- #
# --------------- PRUEBA -------------- #
# ------------------------------------- #
tst = np.loadtxt('./data/OR_tst.csv',delimiter=',')

# ------------ COEFICIENTES ----------- #

yd = [] # Salida esperadas
for i in range(len(tst)): 
    fila = tst[i]
    cant_e = len(fila) - 1 # Cantidad de entradas
    yd.append(fila[cant_e]) # Salidas esperadas
    aux = [-1]
    for j in range(cant_e):
        aux.append(fila[j])
    tst[i] = aux # Vector de entradas

# ------------- ALGORITMO ------------- #
cont_error = 0
for patron in range(len(tst)):
    x = tst[patron] # Entradas
    y = w @ x
    y = np.sign(y)

    error = yd[patron] - y
    if error != 0: cont_error += 1

print('Pesos:',w)
print('Cantidad de errores en la prueba:',cont_error,'/',len(tst))