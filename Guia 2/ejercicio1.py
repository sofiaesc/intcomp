import random
import csv
import numpy as np

# Determino la estructura de mi red neuronal: el número de capas y neuronas para cada capa.
cant_entradas = np.array([2,3,2,1]) # Variable a cambiar según mi red neuronal
# Primer elemento son las entradas x1,x2,... del archivo.
# Próximos elementos me determinan la cantidad de capas y la cantidad de neuronas por capa.
cant_capas = len(cant_entradas) - 1
cant_salidas = cant_entradas[cant_capas]

# ------- DATOS INICIALES ------- #
trn = np.loadtxt('./data/XOR_trn.csv',delimiter=',')

yd = [] # Salida esperadas
for i in range(len(trn)):
    fila = trn[i]
    cant_e = cant_entradas[0] # Cantidad de entradas
    yd.append(fila[cant_e]) # Salidas esperadas
    aux = [-1]
    for j in range(cant_e):
        aux.append(fila[j])
    trn[i] = aux # Vector de entradas por patrón

# Inicializo la matriz de pesos para cada una de las capas.
w = []
for i in range(len(cant_entradas)-1):
    w_aux = np.random.rand(cant_entradas[i+1],cant_entradas[i]+1)-0.5
    w.append(w_aux) # añado al vector de matrices de pesos

# Estructura de la matriz:
# print(w)
# El elemento w_ij = w[i][j] de la matriz me da los pesos correspondientes a la neurona j de la capa i. 
# El elemento w_ij[k] me da el peso asociado a la entrada k de la neurona j de la capa i.

# Definimos los vectores de salidas y deltas (vectores de vectores):
y = np.empty(cant_capas,dtype=object)
delta = np.empty(cant_capas,dtype=object)

# Otros datos para el algoritmo:
epoca = 0
epoca_max = 10 # Máximo de iteraciones
mu = 0.1 # Velocidad de aprendizaje
b = 0.5 # Constante b para sismóidea
perc_error_max = 0.02 # Porcentaje máximo de error
errores_por_epoca = []
mse_por_epoca = []

#----------------------------------------------------------#
# ALGORITMO (ENTRENAMIENTO):
while (epoca < epoca_max): 
    
    #--------------------------------#
    #--------- Aprendizaje ----------#
    #--------------------------------#

    for patron in range(len(trn)):

        # -------- PROPAGACIÓN HACIA ADELANTE: -------- #
        entradas = trn[patron]
        for i in range(cant_capas):
            v = w[i]@entradas # Producto interno de pesos y entradas
            v_a = 2/(1+np.exp(-b*v)) - 1 # Función de activación
            y[i]=v_a # Agrego la salida al vector de salidas
            entradas = np.concatenate(([-1],v_a),axis=None) # Entrada de la próxima capa es la salida de esta capa
        
        # ---------- PROPAGACIÓN HACIA ATRÁS: --------- #
        error = y[-1] - yd[patron]
        # Con el error, obtengo el delta de la capa de salida
        delta[-1]=error*(1/2)*(1+y[-1])*(1-y[-1])
        # Propago ese delta hacia las capas anteriores:
        for i in range(cant_capas-1,0,-1): # Voy de la capa N hasta la 1
            w_i = w[i][:,1:].T
            d = np.dot(w_i,delta[i])
            delta[i-1] = d*(1/2)*(1+y[i-1])*(1-y[i-1])
        
        # ----------- ACTUALIZAR LOS PESOS: ----------- # 
        # !!!!!! FIXME
        entradas = trn[patron]
        for i in range(cant_capas):
            delta_peso = mu*delta[i]*entradas
            w[i] += delta_peso 
            entradas = np.concatenate(([-1],y[i]),axis=None) # entrada para próxima capa es la salida de esta
        
    #--------------------------------#
    #---------- Evaluación ----------#
    #--------------------------------#
    cont_errores = 0    # Contador de errores
    cont_mse = 0        # Contador para error cuadrático medio

    for patron in range(len(trn)): 

        # PROPAGACIÓN HACIA ADELANTE
        entradas = trn[patron]
        for i in range(cant_capas):
            v = w[i]@entradas # Producto interno de pesos y entradas
            v_a = 2/(1+np.exp(-b*v)) - 1 # Función de activación
            y[i]=v_a # Agrego la salida al vector de salidas
            entradas = np.concatenate(([-1],v_a),axis=None) # Entrada de la próxima capa es la salida de esta capa

        # CODIFICACIÓN:
        yc = -1 if(y < 0) else 1

        # Actualizo contadores de error:
        if(yd != yc): cont_errores += 1
        cont_mse += np.sum(np.square(yd-yc)) 

    # Actualizo arrays para grafica de error:
    mse = cont_mse/len(trn)
    mse_por_epoca.append(mse)
    errores_por_epoca.append(cont_errores)
    # Porcentaje de error para criterio de parada:
    perc_error = cont_errores*100/len(trn)
    if(perc_error > perc_error_max): break

    epoca += 1

#----------------------------------------------------------#
# ALGORITMO (TESTING):