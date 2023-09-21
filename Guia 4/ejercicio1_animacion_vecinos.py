import random
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt

filas = 15                       # Filas de neuronas
columnas = 15                    # Columnas de neuronas
v_rv = [8,6,4,2,0]           
inputs = np.loadtxt('./data/circulo.csv',delimiter=',')
cant_e = len(inputs[0])

W = np.empty([filas,columnas], dtype=object)
for i in range(filas):
    for j in range(columnas):
        W[i,j] = np.random.rand(cant_e)-0.5

plt.ion()
fig,ax = plt.subplots()

for _ in range(5):  # Pruebo un punto con las tres etapas y paso a otro punto
    indice = [random.randint(0,filas-1),random.randint(0,columnas-1)]   # Elijo un nodo al azar
    for prueba in range(len(v_rv)): # pruebo para distintos rv
        rv = v_rv[prueba]
    
        ax.clear()
        ax.set(xlim=[-1,filas],ylim=[-1,columnas],title='Neuronas vecinas')
        x_coords, y_coords = [], []
        for x_c in range(filas):
            for y_c in range(columnas):
                x_coords.append(x_c)
                y_coords.append(y_c)
        ax.scatter(x_coords,y_coords,color='black')

        # Me tengo que fijar que no se me vaya a índices negativos o fuera de los límites de la matriz
        for k in range(0,rv+1):           
            d = indice[0] + k                 # Me desplazo una fila abajo
            if (d < filas):
                if(k==0): ax.scatter(indice[0],indice[1],color='yellow')   # Ganadora
                else: ax.scatter(d,indice[1],color='red')
                for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                    dL = indice[1] - j
                    if(dL > -1): ax.scatter(d,dL,color='red')
                    else: break
                for j in range(1,rv+1-k):       # En esa fila, voy a la derecha
                    dR = indice[1] + j
                    if(dR < columnas): ax.scatter(d,dR,color='red')
                    else: break
            else: break
        
        for k in range(1,rv+1): 
            d = indice[0] - k                 # Me desplazo una fila arriba
            if (d > -1):
                ax.scatter(d,indice[1],color='red')
                for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                    dL = indice[1] - j
                    if(dL > -1): ax.scatter(d,dL,color='red')
                    else: break
                for j in range(1,rv+1-k):       # En esa fila, voy a la derecha
                    dR = indice[1] + j
                    if(dR < columnas): ax.scatter(d,dR,color='red')
                    else: break
            else: break
        
        plt.pause(1)

plt.show()
plt.waitforbuttonpress()