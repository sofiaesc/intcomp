import random
import csv
import copy
import time
import numpy as np

def som(inputs,W,max_epocas,v_rv,v_mu,tol):
    W_last = copy.deepcopy(W)
    filas,columnas = W_last.shape
    inicio = time.time()
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

            # ACTUALIZACIONES
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

        W_last = copy.deepcopy(W)
        epoca += 1
    
    fin = time.time()
    print('El entrenamiento finalizó en la época',epoca,'en',round(fin-inicio,2),'segundos.')

    input_neurona = []
    for input in inputs:
        dist = [[np.linalg.norm(input - w) for w in row] for row in W]
        indice = np.argmin(dist)
        input_neurona.append(indice)

    return W,input_neurona