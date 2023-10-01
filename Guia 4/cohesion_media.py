from sklearn.metrics import pairwise_distances
import numpy as np

def cohesion_media(inputs,centroid,input_centroid):
    cm = []
    for j in range(len(centroid)):
        suma = 0
        cont = 0
        for i in range(len(input_centroid)):
            if(input_centroid[i] == j):
                dist = np.linalg.norm(inputs[i]-centroid[j])
                suma += dist
                cont += 1
        if(cont > 0):
            cj = suma/cont
            cm.append(cj)
    
    if len(cm) > 0:
        total = np.mean(cm)
        return total
    return 0