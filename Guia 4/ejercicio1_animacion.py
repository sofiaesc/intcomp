import random
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------- #
# ----------- INICIALIZACIÓN: ------------ #
# ---------------------------------------- #

filas = 3       # Filas de neuronas
columnas = 3    # Columnas de neuronas
rv = 1          # Radio de vecindad

# Entradas desde el archivo .csv:
entradas = np.loadtxt('./data/te.csv',delimiter=',')
cant_entradas = len(entradas[0])

# Inicializo al azar los pesos:
W = np.empty([filas,columnas], dtype=object)
for i in range(filas):
    for j in range(columnas):
        W[i,j] = np.random.rand(cant_entradas)-0.5

# ---------------------------------------- #
# ------------ ENTRENAMIENTO: ------------ #
# ---------------------------------------- #

mu = 0.01          # Velocidad de aprendizaje
epoca = 0
epoca_max = 50
W_last = np.copy(W)

plt.ion()
fig,ax = plt.subplots()

while epoca < epoca_max:

    for input in entradas:

        # Obtengo neurona ganadora buscando peso de menor distancia en W:
        min = float('inf')
        for i in range(len(W)):         # Por cada fila
            for j in range(len(W[0])):  # Por cada elemento de la fila
                dist = np.sum((input-W[i,j])**2)
                if dist < min:
                    min = dist
                    indice = (i,j)      # Índice de la neurona ganadora
        
        #dist = [[np.linalg.norm(input - w) for w in row] for row in W]
        #indice = np.unravel_index(np.argmin(dist),W.shape)

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
    
    # ---------------------------------------- #
    # ---------------- GRÁFICA: -------------- #
    # ---------------------------------------- #
    ax.clear()
    ax.set(xlim=[-1.5,1.5],ylim=[-1.5,1.5],title='Movimientos de las neuronas')
    ax.scatter(entradas[:,0], entradas[:,1], marker='x', color='grey')  # Patrones de entrada

    # Neuronas
    x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
    y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])

    ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')
    hex_codes = []
    with open('./data/hex-codes.txt', 'r') as file:
        for line in file:
            hex_code = line.strip()
            hex_codes.append(hex_code)

    # Conexiones entre neuronas
    for i in range(filas):   
        for j in range(columnas):
            c = hex_codes[(i*(j+2)+10)]
            indice = [i,j]
            w1 = W[i,j]
            for k in range(1,rv+1):           
                d = indice[0] + k                 # Me desplazo una fila abajo
                if (d < filas):
                    w2 = W[d,indice[1]]
                    ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
                    for j in range(1,rv+1):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1):
                            w2 = W[d,dL]
                            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
                    for j in range(1,rv+1):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas):
                            w2 = W[d,dR]
                            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
            
            for k in range(1,rv+1): 
                d = indice[0] - k                 # Me desplazo una fila arriba
                if (d > -1):
                    w2 = W[d,indice[1]]
                    ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
                    for j in range(1,rv+1):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1):
                            w2 = W[d,dL]
                            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
                    for j in range(1,rv+1):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas):
                            w2 = W[d,dR]
                            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
    plt.pause(0.3)

    if(np.array_equal(W,W_last)):        # Si ya no se actualizan los pesos, corto.
        break
    W_last = np.copy(W)

    epoca += 1

plt.show()
plt.waitforbuttonpress()