import random
import csv
import numpy as np

# Determino la estructura de mi red neuronal: el número de capas y neuronas para cada capa.
cant_entradas = np.array([2,3,2,1]) # Variable a cambiar según mi red neuronal
# Primer elemento son las entradas x1,x2,... del archivo.
# Próximos elementos me determinan la cantidad de capas y la cantidad de neuronas por capa.
cant_capas = len(cant_entradas) - 1
cant_salidas = cant_entradas[cant_capas]


#----------------------------------------------------------#
#---------------------ENTRENAMIENTO------------------------#
#----------------------------------------------------------#

# ----> DATOS DE ENTRADA
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

# ----> DATOS PARA EL ALGORITMO:
epoca = 0               # Contador para época actual
epoca_max = 50          # Máximo de iteraciones
mu = 0.01               # Velocidad de aprendizaje
b = 1                   # Constante b para sismóidea
perc_error_max = 0.05   # Porcentaje máximo de error
errores_por_epoca = []
mse_por_epoca = []

# ----> ALGORITMO:
while (epoca < epoca_max):  
    
    #--------------------------------#
    #--------- Aprendizaje ----------#
    #--------------------------------#

    for patron in range(len(trn)):

        # PROPAGACIÓN HACIA ADELANTE: Obtengo la salida de las capas y las propago como entradas de las próximas
        entradas = trn[patron]          # La primera capa tiene las entradas en el archivo .csv
        for i in range(cant_capas):
            v = w[i]@entradas                  # Producto interno de pesos y entradas
            y[i] = 2/(1+np.exp(-b*v))-1        # Salida con función de activación
            entradas = np.hstack((-1,y[i]))    # Entrada de la próxima capa es la salida de esta capa
        
        # PROPAGACIÓN HACIA ATRÁS: Obtengo el delta de la capa de salida y lo propago a las capas anteriores
        error = yd[patron] - y[-1]                    
        delta[-1]=error*(1/2)*(1+y[-1])*(1-y[-1])       # Con el error, obtengo el delta de la capa de salida
        for i in range(cant_capas-1,0,-1):
            w_i = w[i][:,1:].T                          # No tomo el peso w0 (umbral) porque no tiene delta para propagar
            d = np.dot(w_i,delta[i])
            delta[i-1] = d*(1/2)*(1+y[i-1])*(1-y[i-1])  # Con los pesos de la capa i obtenemos el delta de la capa i-1
        
        # ACTUALIZAR LOS PESOS: Ajusto los pesos con la velocidad de aprendizaje, la entrada y su delta.
        entradas = trn[patron]          # La primera capa tiene las entradas en el archivo .csv
        for i in range(cant_capas):
            delta_peso = mu*(np.outer(delta[i],entradas))
            w[i] += delta_peso
            entradas = np.hstack((-1,y[i]))    # Entrada para próxima capa es la salida de esta
        
    #--------------------------------#
    #---------- Evaluación ----------#
    #--------------------------------#
    cont_errores = 0    # Contador de errores
    cont_mse = 0        # Contador para error cuadrático medio

    for patron in range(len(trn)): 

        # PROPAGACIÓN HACIA ADELANTE: Obtengo la salida de las capas y las propago como entradas de las próximas
        entradas = trn[patron]          # La primera capa tiene las entradas en el archivo .csv
        for i in range(cant_capas):
            v = w[i]@entradas                  # Producto interno de pesos y entradas
            y[i] = 2/(1+np.exp(-b*v)) - 1      # Salida con función de activación
            entradas = np.hstack((-1,y[i]))    # Entrada de la próxima capa es la salida de esta capa

        # CODIFICACIÓN: Función signo
        if (y[-1] < 0): yc = -1
        else: yc = 1

        # Actualizo contadores de error:
        if(yd[patron] != yc): cont_errores += 1
        cont_mse += np.sum(np.square(yd[patron]-yc)) 

    # Actualizo arrays para grafica de error:
    mse = cont_mse/len(trn)
    mse_por_epoca.append(mse)
    errores_por_epoca.append(cont_errores)
    # Porcentaje de error para criterio de parada:
    perc_error = cont_errores*100/len(trn)
    if(perc_error < perc_error_max): break

    epoca += 1

print('Finalizó el entrenamiento en la época ',epoca,' con ',cont_errores,'/',len(trn),' errores')

#----------------------------------------------------------#
#------------------------TESTING---------------------------#
#----------------------------------------------------------#

# ----> DATOS DE ENTRADA:
tst = np.loadtxt('./data/XOR_tst.csv',delimiter=',')

yd = []                         # Salidas esperadas
for i in range(len(tst)):
    fila = tst[i]
    cant_e = cant_entradas[0]   # Cantidad de entradas
    yd.append(fila[cant_e])     
    aux = [-1]                  # Añado entrada -1 correspondiente al umbral/sesgo
    for j in range(cant_e):
        aux.append(fila[j])
    tst[i] = aux                # Vector de entradas por patrón

# ----> ALGORITMO:
y = np.empty(cant_capas,dtype=object)   # Vector de salidas
cont_errores = 0                        # Contador de errores

for patron in range(len(tst)): 
    entradas = tst[patron]
    for i in range(cant_capas):
        v = w[i]@entradas                   # Producto interno de pesos y entradas
        y[i] = 2/(1+np.exp(-b*v)) - 1       # Salida con función de activación
        entradas = np.hstack((-1,y[i]))     # La entrada de la próxima capa es la salida de la actual.

    if (y[-1] < 0): yc = -1
    else: yc = 1
    if(yd[patron] != yc): cont_errores += 1

print('Finalizó la prueba con ',cont_errores,'/',len(tst),' errores.')