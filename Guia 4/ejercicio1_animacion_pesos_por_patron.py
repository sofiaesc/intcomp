import random
import csv
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------- #
# ----------- INICIALIZACIÓN: ------------ #
# ---------------------------------------- #

filas = 6                      # Filas de neuronas
columnas = 6                   # Columnas de neuronas
v_rv = [2,1,0]                  # Radio de vecindad
v_epoca_max = [50,50,50]    # Épocas máximas por etapa 
                            # Como este código es para ver la animación y no calcular realmente, le pongo pocas épocas máximas para que se vea bien el gif.
v_mu = [0.05, 0.02, 0.01]         # Velocidad de aprendizaje
tol = 1e-3

# Entradas desde el archivo .csv:
inputs = np.loadtxt('./data/circulo.csv',delimiter=',')
cant_e = len(inputs[0])

# Inicializo al azar los pesos:
W = np.empty([filas,columnas], dtype=object)
for i in range(filas):
    for j in range(columnas):
        W[i,j] = np.random.rand(cant_e)-0.5

# ---------------------------------------- #
# ------------ ENTRENAMIENTO: ------------ #
# ---------------------------------------- #

plt.ion()
fig,ax = plt.subplots()
# Grafico estado inicial con inicialización al azar:
ax.set(xlim=[-1.5,1.5],ylim=[-1.5,1.5],title='Movimientos de las neuronas')
ax.scatter(inputs[:,0], inputs[:,1], marker='x', color='grey')  # Patrones de entrada
x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])
ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')

c = 'black'
for i in range(filas):   
    for j in range(columnas):
        indice = [i,j]
        w1 = W[i,j]

        d1 = indice[0] + 1          # Me desplazo una fila abajo
        if(d1 < filas):                
            w2 = W[d1,indice[1]]
            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
        
        d2 = indice[0] - 1          # Me desplazo una fila arriba
        if(d2 > -1):                
            w2 = W[d2,indice[1]]
            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)

        d3 = indice[1] - 1          # Me desplazo una columna a la izquierda
        if(d3 > -1):
            w2 = W[indice[0],d3]
            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)

        d4 = indice[1] + 1          # Me desplazo una columna a la derecha
        if(d4 < columnas):
            w2 = W[indice[0],d4]
            ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
plt.waitforbuttonpress()
plt.pause(0.1)


for etapa in range(3): # tres etapas: ordenamiento global, transición y ajuste fino (convergencia)
    rv = v_rv[etapa]
    mu = v_mu[etapa]
    epoca = 0
    W_saved = [copy.deepcopy(W)]
    inicio=time.time()

    while epoca < v_epoca_max[etapa]:

        for input in inputs:
            # Obtengo neurona ganadora buscando peso de menor distancia en W:
            dist = [[np.linalg.norm(input - w) for w in row] for row in W]
            indice = np.unravel_index(np.argmin(dist),W.shape)

            # ACTUALIZACIONES
            error = input - W[indice]
            inc = mu*error
            
            # Me tengo que fijar que no se me vaya a índices negativos o fuera de los límites de la matriz
            for k in range(0,rv+1):                 # Empieza en 0 para hacer la fila de la neurona ganadora      
                d = indice[0] + k                   # Me desplazo una fila abajo
                if (d < filas):
                    W[d,indice[1]] = W[d,indice[1]] + inc
                    for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1): W[d,dL] = W[d,dL] + inc
                        else: break
                    for j in range(1,rv+1-k):
                        dR = indice[1] + j          # En esa fila, voy a la derecha
                        if(dR < columnas): W[d,dR] = W[d,dR] + inc
                        else: break
                else: break
            
            for k in range(1,rv+1):               # Empieza en 1 porque ya hice la fila de la neurona ganadora antes
                d = indice[0] - k                 # Me desplazo una fila arriba
                if (d > -1):
                    W[d,indice[1]] = W[d,indice[1]] + inc
                    for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1): W[d,dL] = W[d,dL] + inc  
                        else: break  
                    for j in range(1,rv+1-k): 
                        dR = indice[1] + j          # En esa fila, voy a la derecha
                        if(dR < columnas): W[d,dR] = W[d,dR] + inc
                        else: break
                else: break
            
            # ---------------------------------------- #
            # ---------------- GRÁFICA: -------------- #
            # ---------------------------------------- #
            ax.clear()
            ax.set(xlim=[-1.5,1.5],ylim=[-1.5,1.5],title='Movimientos de las neuronas')
            ax.scatter(inputs[:,0], inputs[:,1], marker='x', color='grey')  # Patrones de entrada

            # Neuronas
            x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
            y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])
            ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')

            # Conexiones entre neuronas:
            c = 'black'
            for i in range(filas):   
                for j in range(columnas):
                    indice = [i,j]
                    w1 = W[i,j]

                    d1 = indice[0] + 1          # Me desplazo una fila abajo
                    if(d1 < filas):                
                        w2 = W[d1,indice[1]]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
                    
                    d2 = indice[0] - 1          # Me desplazo una fila arriba
                    if(d2 > -1):                
                        w2 = W[d2,indice[1]]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)

                    d3 = indice[1] - 1          # Me desplazo una columna a la izquierda
                    if(d3 > -1):
                        w2 = W[indice[0],d3]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)

                    d4 = indice[1] + 1          # Me desplazo una columna a la derecha
                    if(d4 < columnas):
                        w2 = W[indice[0],d4]
                        ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color=c)
            
            plt.waitforbuttonpress()
            plt.pause(0.1)
    
        
        
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

    fin = time.time()
    duracion = fin - inicio 
    print('La etapa',etapa+1,'finalizó en la época',epoca,'en',round(duracion,2),'segundos.')

plt.show()
plt.waitforbuttonpress()