import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def kmedias(k,inputs,max_epoca):
    random.seed()   # Para que se inicialicen igual con los distintos k así comparo justamente.

    # Inicialización de centroides
    c = np.empty(k,dtype=object)
    for i in range(k):
        c[i] = random.choice(inputs)              # Patrón de entrada al azar
    
    epoca = 1
    last_input_centroid = np.empty(len(inputs))     # Vector donde guardo la última asignación de centroides a patrones
    input_centroid = np.empty(len(inputs))          # Vector donde guardo la asignación actual de centroides a patrones

    while epoca <= max_epoca:
        flag = False
    
        for i in range(len(inputs)):                # Asigno centroides a patrones:
            dist = [np.linalg.norm(inputs[i] - c[j]) for j in range(k)]
            if (np.argmin(dist) != input_centroid[i]):
                input_centroid[i] = np.argmin(dist)
                flag = True     # Si reasigné, pongo la bandera en true.

        for j in range(k):                          # Recalculo coordenadas de centroides
            index_inputs_cj = np.where(input_centroid == j)[0]
            if(len(index_inputs_cj) != 0):          # Evito dividir por cero en el promedio
                suma = sum(inputs[i] for i in index_inputs_cj)
                c[j] = suma/len(index_inputs_cj)

        if (flag == False):     # Si no reasigné nunca, corto.
            break

        epoca += 1

    return inputs,c,input_centroid    # Devuelvo patrones de entrada, los centroides y los centroides que le corresponden a cada entrada