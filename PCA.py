
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA

D = int(raw_input())
N = int(raw_input())
distance_type =int(raw_input())
X = int(raw_input())

P_data = [int(each) for each in raw_input().split()]

other_patient_data = []
for i in range(N):
    other_patient_data.append([int(each) for each in raw_input().split()])

def get_distance(distance_type, a, b):
    if distance_type == 1: # Manhattan distance
        return distance.minkowski(a, b, p=1)

    elif distance_type == 2: # Euclidean distance
        return distance.minkowski(a, b, p=2)

    elif distance_type == 3:
        return distance.minkowski(a, b, p=float('inf'))

    else:
        return distance.cosine(a, b)

def get_pca_vectors(number_components, P_data, other_patient_data):
    train_data = np.array([P_data] + other_patient_data)
    pca = PCA(n_components=number_components)
    pca.fit(train_data)

    P_data = pca.transform([P_data])

    new_other_data = []
    for each in other_patient_data:
        new_other_data.append(list(list(pca.transform([each]))[0]))

    return P_data, new_other_data, pca.explained_variance_



def get_index_list(P_data, other_patient_data):
    distance_list = []
    if X != -1:
        P_data, other_patient_data, variance = get_pca_vectors(X, P_data, other_patient_data)

    for i in range(0, len(other_patient_data)):
        distance_list.append((i, get_distance(distance_type, P_data, other_patient_data[i])))

    distance_list = sorted(distance_list, key=lambda x: x[1])[0:5]

    for each in distance_list:
        print each[0]+1

    if X!=-1:
        print sum(variance)


get_index_list(P_data, other_patient_data)
