import random
import csv
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

def som_grafica(inputs,W,max_epocas,v_rv,v_mu,tol):
    W_last = copy.deepcopy(W)
    filas,columnas=W.shape

    plt.ion()
    fig,ax = plt.subplots()
    # Grafico estado inicial con inicialización al azar:
    ax.scatter(inputs[:,0], inputs[:,1], marker='x', color='grey')  # Patrones de entrada
    x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
    y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])
    ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')

    for i in range(filas):   
        for j in range(columnas):
            w1 = W[i,j]
            d1 = i + 1          # Me desplazo una fila abajo
            if(d1 < filas):                
                w2 = W[d1,j]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
            d2 = i - 1          # Me desplazo una fila arriba
            if(d2 > -1):                
                w2 = W[d2,j]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
            d3 = j - 1          # Me desplazo una columna a la izquierda
            if(d3 > -1):
                w2 = W[i,d3]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
            d4 = j + 1          # Me desplazo una columna a la derecha
            if(d4 < columnas):
                w2 = W[i,d4]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
    plt.waitforbuttonpress()
    plt.pause(0.1)

    for epoca in range(max_epocas):
        for input in inputs:
            # Decrecimiento de los parámetros para etapa de transición:
            step = (v_mu[0] - v_mu[1])/max_epocas
            mu = v_mu[0] - epoca*step             # Si v_mu[0] = v_mu[1], step = 0 y mu no se modifica.
            step = (v_rv[0] - v_rv[1])/max_epocas
            rv = round(v_rv[0] - epoca*step)      # Lo mismo para rv.

            # Obtengo neurona ganadora buscando peso de menor distancia en W:
            dist = [[np.linalg.norm(input - w) for w in row] for row in W]
            indice = np.unravel_index(np.argmin(dist),W.shape)

            # Grafico de nuevo con entrada actual y neurona ganadora
            ax.clear()
            ax.scatter(inputs[:,0], inputs[:,1], marker='x', color='grey')  # Patrones de entrada
            ax.scatter(input[0],input[1],marker='x',color='red')   # Patron de entrada actual

            x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
            y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])
            ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')
            ax.scatter(W[indice][0],W[indice][1],color='yellow')    # Neurona ganadora

            for i in range(filas):   
                for j in range(columnas):
                    w1 = W[i,j]
                    d1 = i + 1          # Me desplazo una fila abajo
                    if(d1 < filas):                
                        w2 = W[d1,j]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
                    d2 = i - 1          # Me desplazo una fila arriba
                    if(d2 > -1):                
                        w2 = W[d2,j]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
                    d3 = j - 1          # Me desplazo una columna a la izquierda
                    if(d3 > -1):
                        w2 = W[i,d3]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
                    d4 = j + 1          # Me desplazo una columna a la derecha
                    if(d4 < columnas):
                        w2 = W[i,d4]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
            plt.waitforbuttonpress()
            plt.pause(0.1)

            # Me tengo que fijar que no se me vaya a índices negativos o fuera de los límites de la matriz
            for k in range(0,rv+1):                 # Empieza en 0 para hacer la fila de la neurona ganadora      
                d = indice[0] + k                   # Me desplazo una fila abajo
                if (d < filas):
                    W[d,indice[1]] += mu*(input - W[d,indice[1]])
                    for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1): W[d,dL] += mu*(input - W[d,dL])
                        else: break
                    for j in range(1,rv+1-k):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas): W[d,dR] += mu*(input - W[d,dR])
                        else: break
                else: break
            
            for k in range(1,rv+1):               # Empieza en 1 porque ya hice la fila de la neurona ganadora antes
                d = indice[0] - k                 # Me desplazo una fila arriba
                if (d > -1):
                    W[d,indice[1]] += mu*(input - W[d,indice[1]])
                    for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1): 
                            W[d,dL] += mu*(input - W[d,dL])
                        else: break
                    for j in range(1,rv+1-k):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas): W[d,dR] += mu*(input - W[d,dR])
                        else: break
                else: break
        
            # Grafico de nuevo después de actualizar los pesos con entrada actual y neurona ganadora
            ax.clear()
            ax.scatter(inputs[:,0], inputs[:,1], marker='x', color='grey')  # Patrones de entrada
            ax.scatter(input[0],input[1],marker='x',color='red')   # Patron de entrada actual

            x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
            y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])
            ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')
            ax.scatter(W[indice][0],W[indice][1],color='yellow')    # Neurona ganadora

            for i in range(filas):   
                for j in range(columnas):
                    w1 = W[i,j]
                    d1 = i + 1          # Me desplazo una fila abajo
                    if(d1 < filas):                
                        w2 = W[d1,j]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
                    d2 = i - 1          # Me desplazo una fila arriba
                    if(d2 > -1):                
                        w2 = W[d2,j]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
                    d3 = j - 1          # Me desplazo una columna a la izquierda
                    if(d3 > -1):
                        w2 = W[i,d3]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
                    d4 = j + 1          # Me desplazo una columna a la derecha
                    if(d4 < columnas):
                        w2 = W[i,d4]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
            plt.waitforbuttonpress()
            plt.pause(0.1)

        # Analizo convergencia para ver si corto antes del máximo de épocas:
        flag = False
        for i in range(filas):
            for j in range(columnas):
                if not np.all(np.abs(W_last[i,j] - W[i,j]) < tol):
                    flag = True  # Si hay alguna diferencia mayor que la tolerancia, pongo bandera en true y continúo con las épocas
        if flag == False:
            break

        W_last = copy.deepcopy(W)
        epoca += 1

    plt.show()
    plt.waitforbuttonpress()

# ESTRUCTURA DE LA RED SOM (cambio estos argumentos para ver cómo cambia el comportamiento)
filas = 5          # Filas de neuronas
columnas = 5       # Columnas de neuronas
tol = 1e-8         # Tolerancia de error para indicar convergencia
mu = [0.5,0.5]     # Velocidad de aprendizaje inicial y final
rv = [2,2]         # Radio de activación inicial y final
max_epocas = 10    # No importa, vemos sólo evolución por patrones

inputs = np.loadtxt('./data/te.csv',delimiter=',')   

pesos = 0
if(pesos == 1):     # Pesos inicializados al azar
    W = np.empty([filas,columnas], dtype=object)            
    for i in range(filas):
        for j in range(columnas):
            W[i,j] = np.random.rand(len(inputs[0]))-0.5
else:               # Pesos en grilla
    x_grid = np.linspace(1, 2, columnas)
    y_grid = np.linspace(1, 2, filas)
    W = np.empty([filas, columnas], dtype=object)
    for i in range(filas):
        for j in range(columnas):
            W[i, j] = [x_grid[j], y_grid[i]]

W = som_grafica(inputs,W,max_epocas,rv,mu,tol)     