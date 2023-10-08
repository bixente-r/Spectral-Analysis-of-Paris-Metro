# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = {
    '1': 'yellow',
    '2': 'blue',
    '3': 'olive',
    '3bis': 'cyan',
    '4': 'mediumorchid',
    '5': 'orange',
    '6': 'palegreen',
    '7': 'pink',
    '7b2': 'pink',
    '7bis': 'mediumseagreen',
    '8': 'plum',
    '9': 'yellowgreen',
    '10': 'gold',
    '10b2': 'gold',
    '11': 'brown',
    '12': 'green',
    '13': 'lightblue',
    '13b2': 'lightblue',
    '14': 'rebeccapurple',
    'A': 'red'
}
colors2 = {
    '1': 'yellow',
    '2': 'blue',
    '3': 'olive',
    '3bis': 'cyan',
    '4': 'mediumorchid',
    '5': 'orange',
    '6': 'palegreen',
    '7': 'pink',
    '7bis': 'mediumseagreen',
    '8': 'plum',
    '9': 'yellowgreen',
    '10': 'gold',
    '11': 'brown',
    '12': 'green',
    '13': 'lightblue',
    '14': 'rebeccapurple',
    'A': 'red'
}

# Obtenir la liste des stations avec >= i correspondances
# (i=1 : une ligne et  pas de correspondances ; i=2 : deux lignes et une correspondance)
def get_list_corr(i):
    df = pd.read_csv('metro_paris_par_station_13_04.csv')
    df = df.drop("Num", axis=1)

    df = df.drop("Lon X", axis=1)
    df = df.drop("Lat Y", axis=1)
    df = df.drop("X", axis=1)
    df = df.drop("Y", axis=1)
    df = df.drop("B", axis=1)
    df = df.drop("C", axis=1)
    df = df.drop("D", axis=1)
    df = df.drop("E", axis=1)
    df = df.drop(308, axis=0)
    df = df.drop(309, axis=0)
    df = df.drop(310, axis=0)
    df = df.drop(311, axis=0)
    df = df.drop(312, axis=0)
    df = df.drop(313, axis=0)
    df = df.drop(314, axis=0)

    df.fillna(0, inplace = True)            #Remplace les valeurs vide "NAN" par 0

    #print(df)
    df = df.drop("Nom", axis=1)
    df = df.drop("MATRICE", axis=1)
    list  = df[df.sum(axis=1) >= i].index.tolist()
    list = [i+1 for i in list]
    return list

# MATRICE D'ADJACENCE
df = pd.read_csv("Metro_parisien.csv", sep=',')
df.fillna(0, inplace = True)
df = df.astype('int')

for i in df.columns:
    for index, valeur in enumerate(df[i]):
        if df.loc[index, i] != 0:
            df.loc[index, i] = 308 + 1 - valeur



# create a list from every column in the DataFrame without NaN values
line_indices_list = []

for col in df.columns:
    col_list = df[col].tolist()
    col_list = [i for i in col_list if i != 0]
    line_indices_list.append(col_list)


print(line_indices_list)

"""del line_indices_list[7][0]
line_indices_list[7] = line_indices_list[7] + line_indices_list[8]
del line_indices_list[8]
del line_indices_list[11][0]
line_indices_list[10] = line_indices_list[10] + line_indices_list[11]
del line_indices_list[11]
del line_indices_list[14][0]
line_indices_list[13] = line_indices_list[13] + line_indices_list[14]
del line_indices_list[14]"""

max_station = max([max(line) for line in line_indices_list]) 
print(max_station)

# Initialize an adjacency matrix with zeros
A = np.zeros((max_station, max_station))

for line in line_indices_list:
    for k in range(len(line)):
        if k == 0:                              # If the station is at the beginning of the line, it is connected to the next station on the line
            A[line[k]-1, line[k + 1]-1] = 1
            A[line[k + 1]-1, line[k]-1] = 1
        elif k == len(line)-1:                  # If the station is at the end of the line, it is connected to the previous station on the line
            A[line[k]-1, line[k - 1]-1] = 1
            A[line[k - 1]-1, line[k]-1] = 1
        else:                                   # If the station is in the middle of a branch, it is connected to the previous and next stations on the line
            A[line[k]-1, line[k - 1]-1] = 1
            A[line[k - 1]-1, line[k]-1] = 1
            A[line[k]-1, line[k + 1]-1] = 1
            A[line[k + 1]-1, line[k]-1] = 1

print(f"Taille de la matrice d'adjacence: {A.shape}")

is_symmetric = np.allclose(A, A.T)

if is_symmetric:
    print("La matrice est symétrique")
else:
    print("La matrice n'est pas symétrique")

print(f"Le nombre d'arrêtes est : {np.sum(A == 1)}")

"""# Retirer list(id_station) à A 
def remove_stat(rm):
    df = pd.read_csv("Metro_parisien.csv", sep=',')
    df.fillna(0, inplace = True)
    df = df.astype('int')

    for i in rm:
        for col in df.columns:
            # replace all elements equal to 2 with NaN in current column
            df[col] = df[col].replace(i, 0)

    # create a list from every column in the DataFrame without NaN values
    line_indices_list = []

    for col in df.columns:
        col_list = df[col].tolist()
        col_list = [i for i in col_list if i != 0]
        line_indices_list.append(col_list)

    print(line_indices_list)


    max_station = max([max(line) for line in line_indices_list]) 
    print(max_station)

    # Initialize an adjacency matrix with zeros
    A = np.zeros((max_station, max_station))

    for line in line_indices_list:
        for k in range(len(line)):
            if k == 0:                              # If the station is at the beginning of the line, it is connected to the next station on the line
                A[line[k]-1, line[k + 1]-1] = 1
                A[line[k + 1]-1, line[k]-1] = 1
            elif k == len(line)-1:                  # If the station is at the end of the line, it is connected to the previous station on the line
                A[line[k]-1, line[k - 1]-1] = 1
                A[line[k - 1]-1, line[k]-1] = 1
            else:                                   # If the station is in the middle of a branch, it is connected to the previous and next stations on the line
                A[line[k]-1, line[k - 1]-1] = 1
                A[line[k - 1]-1, line[k]-1] = 1
                A[line[k]-1, line[k + 1]-1] = 1
                A[line[k + 1]-1, line[k]-1] = 1

    print(f"Taille de la matrice d'adjacence: {A.shape}")

    is_symmetric = np.allclose(A, A.T)

    if is_symmetric:
        print("La matrice est symétrique")
    else:
        print("La matrice n'est pas symétrique")

    print(f"Le nombre d'arrêtes est : {np.sum(A == 1)}")
    return A"""

# Retirer list(id_station) à A 
def remove_stat(A,rm):

    arr = A.copy()
    for i in rm:
        arr = np.delete(np.delete(arr, i-1, axis=0), i-1, axis=1)

    is_symmetric = np.allclose(arr, arr.T)

    if is_symmetric:
        print("La matrice est symétrique")
    else:
        print("La matrice n'est pas symétrique")

    print(f"Le nombre d'arrêtes est : {np.sum(arr == 1)}")
    return arr

# Obtenir D, L et L_norm de A
def get_matrix(A):
    # Calcul de la matrice des degrés
    D = np.diag(np.sum(A, axis=1))
    # check if there is a 0 on the diagonal
    has_zero = False
    for i in range(D.shape[0]):
        if D[i, i] == 0:
            has_zero = True
            print(i)
            break

    if has_zero:
        print("The matrix has a 0 on its diagonal.")
    else:
        print("The matrix does not have a 0 on its diagonal.")

    print("Matrice d'adjacence :")
    print(A)

    print("Matrice des degrés :")
    print(D)
    print("Trace de la matrice des degrés :")
    print(np.trace(D))

    # Calcul de la matrice Laplacienne
    L = D - A

    # Calcul de la matrice Laplacienne normalisée
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))
    # D_sqrt_inv = np.diag(1/np.sqrt(np.sum(A, axis=1)))
    L_norm = np.dot(D_sqrt_inv, np.dot(L, D_sqrt_inv))

    #print("La matrice Laplacienne normalisée est :")
    #print(L_norm)
    return D, L, L_norm

D, L, L_norm = get_matrix(A)

# Vérifier les propriétés des Matrices
def is_complet(M):
    n = max_station
    res = np.sum(M == 1) == n*(n-1)/2
    return res


def verified_LP(M,A):
    lambdas = sorted(np.linalg.eigvals(M))
    lambda0 = min(lambdas)
    n = len(lambdas)
    res1 = (lambda0 < 1e-14 and lambda0 > -1e-14)               #Check if the lower eigen value equal 0
    res2 = (sum(lambdas) <= 308 + 1e15)                         #Check if the sum of eigen velues equal 308 (bc connex)
    for i in range(n):                                          #Check if every eigen value is lwoer or equal to 2
        if lambdas[i] > 2:
            res3 = False
        else:
            res3 = True
    if is_complet(A):
        res4 = lambdas[1] == n/(n-1)
    else:
        res4 = lambdas[1] <= n/(n-1)
    res = res1 and res2 and res3 and res4
    if res:
        return "La matrice est Laplacienne Normalisée."
    else:
        return "La matrice n'est pas Laplacienne Normalisée."

print(is_complet(A))
verified_LP(L_norm, A)

A2 = remove_stat(A,[15])
D2, L2, L_norm2 = get_matrix(A2)

spectre = np.linalg.eigvals(L_norm)
spectre2 = np.linalg.eigvals(L_norm2)
line_ids = get_list_corr(5)
x = [i for i in range(1,309)]

coefficients = np.polyfit(x, spectre, 1)  # 1 indique le degré de la régression (une droite)

# Extraction des coefficients
pente = coefficients[0]
ordonnee_origine = coefficients[1]

regression_y = pente * np.array(x) + ordonnee_origine

import seaborn as sns
plt.figure(1)
sns.histplot(spectre, alpha=0.5, bins=60)
plt.title('Repartition of eigenvalues ')
plt.xlabel('value of eigenvalues')
plt.ylabel('Number of eigenvalues')

import seaborn as sns
plt.figure(1)
sns.histplot(spectre, alpha=0.5, bins=60, color='blue', label='base')
sns.histplot(spectre2, alpha=0.5, bins=60, color='red', label='mod')
plt.legend()
plt.title('Repartition of eigenvalues ')
plt.xlabel('value of eigenvalues')
plt.ylabel('Number of eigenvalues')


plt.figure(2)
plt.title('Représentation des valeurs propres par ligne')
for l in range(len(line_indices_list)):
    for i in line_indices_list[l]:
        plt.plot(i,spectre[i-1], marker='.', markerfacecolor=list(colors.values())[l], markeredgecolor=list(colors.values())[l])
for i in line_ids:
    plt.plot(i, spectre[i-1],marker='.', markerfacecolor='black', markeredgecolor='black', markersize=8)
plt.show()

plt.figure()
plt.plot(x,regression_y)
plt.show
print(f"pente de la droite : {pente}")

plt.figure(2)
plt.title('Valeurs propre ligne 1')
for i in line_indices_list[-4]:
    plt.plot(i,spectre[i-1], marker='.', color='gold')

corr_vp = []
for i in line_ids:
    corr_vp.append(spectre[i-1])

print(np.mean(corr_vp))

list_list_vp = []

for l in range(len(line_indices_list)):
    a = []
    for i in line_indices_list[l]:
        a.append(spectre[i-1])
    list_list_vp.append(a)

list_list_vp[2] = list_list_vp[2] + list_list_vp[3]
del list_list_vp[3]
list_list_vp[6] = list_list_vp[6] + list_list_vp[7]
del list_list_vp[7]
list_list_vp[10] = list_list_vp[10] + list_list_vp[11]
del list_list_vp[11]
list_list_vp[13] = list_list_vp[13] + list_list_vp[14]
del list_list_vp[14]

for i in range(len(list_list_vp)):
    plt.plot(i+1, np.mean(list_list_vp[i]), marker='.', markerfacecolor=list(colors2.values())[i], markeredgecolor=list(colors2.values())[i])



plt.figure(1)
plt.title('Graphe des valeurs propres')
plt.plot(np.linalg.eigvals(L_norm))#, marker='.', markerfacecolor='green', markeredgecolor='green')

plt.figure(2)
plt.title('Représentation des valeurs propres en fonction de la ligne')
for l in range(len(line_indices_list)):
    for i in line_indices_list[l]:
        plt.plot(i,spectre2[i-1], marker='.', markerfacecolor=list(colors.values())[l], markeredgecolor=list(colors.values())[l])
for i in line_ids:
    plt.plot(i, spectre2[i-1],marker='.', markerfacecolor='black', markeredgecolor='black', markersize=9)
plt.show()

# Échange des valeurs de tous les éléments de la matrice
n = len(A)
for i in range(n):
    for j in range(i, n):
        A[i][j], A[j][i] = A[j][i], A[i][j]

D, L, L_norm = get_matrix(A)

spectre = np.linalg.eigvals(L_norm)

plt.figure(2)
plt.title('Représentation des valeurs propres en fonction de la ligne')
for l in range(len(line_indices_list)):
    for i in line_indices_list[l]:
        plt.plot(i,spectre[i-1], marker='.', markerfacecolor=list(colors.values())[l], markeredgecolor=list(colors.values())[l])
plt.show()

