import random
import csv
import copy
import numpy as np

def som(filas,columnas,rv,cant_e,archivo):
    # ---------------------------------------- #
    # ----------- INICIALIZACIÓN: ------------ #
    # ---------------------------------------- #

    filas = 5
    columnas = 5
    rv = 2
    cant_e = 4
    archivo = './data/irisbin_trn.csv'

    # Levanto datos del archivo
    trn = np.loadtxt(archivo,delimiter=',')
    inputs = np.empty(len(trn),dtype=object)      # Vector de entradas

    for i in range(len(trn)):
        fila = trn[i]
        inputs[i] = fila[0:cant_e]

    # Inicializo al azar los pesos:
    W = np.empty([filas,columnas], dtype=object)
    for i in range(filas):
        for j in range(columnas):
            W[i,j] = np.random.rand(cant_e)-0.5

    # ---------------------------------------- #
    # ------------ ENTRENAMIENTO: ------------ #
    # ---------------------------------------- #
    mu = 0.01          # Velocidad de aprendizaje
    epoca = 0
    epoca_max = 50
    tol = 1e-4
    W_saved = [copy.deepcopy(W)]

    while epoca < epoca_max:

        for input in inputs:

            # Obtengo neurona ganadora buscando peso de menor distancia en W:
            dist = [[np.linalg.norm(input - w) for w in row] for row in W]
            indice = np.unravel_index(np.argmin(dist),W.shape)

            # ACTUALIZACIONES
            error = input - W[indice]
            inc = mu*error
            W[indice] += inc
            
            # Actualización de los k vecinos:
            # Me tengo que fijar que no se me vaya a índices negativos o fuera de los límites de la matriz
            for k in range(1,rv+1):           
                d = indice[0] + k                 # Me desplazo una fila abajo
                if (d < filas):
                    W[d,indice[1]] += inc
                    for j in range(1,rv+1):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1):
                            W[d,dL] += inc
                    for j in range(1,rv+1):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas):
                            W[d,dR] += inc
            
            for k in range(1,rv+1): 
                d = indice[0] - k                 # Me desplazo una fila arriba
                if (d > -1):
                    W[d,indice[1]] += inc
                    for j in range(1,rv+1):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1):
                            W[d,dL] += inc
                    for j in range(1,rv+1):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas):
                            W[d,dR] += inc
        
        flag = False
        for i in range(filas):
            for j in range(columnas):
                W_last = W_saved[-1]
                if not np.all(np.abs(W_last[i,j] - W[i,j]) < tol):
                    flag = True  # Si hay alguna diferencia mayor que la tolerancia, pongo bandera en true y continúo con las épocas
        if flag == False:
            break

        W_saved.append(copy.deepcopy(W))

        epoca += 1

    input_neurona = []
    for input in inputs:
        dist = [[np.linalg.norm(input - w) for w in row] for row in W]
        indice = np.argmin(dist)
        input_neurona.append(indice)

    return inputs,input_neurona