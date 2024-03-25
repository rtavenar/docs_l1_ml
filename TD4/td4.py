from sklearn.datasets import make_blobs, fetch_olivetti_faces
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np



def charger_donnees():
    """Charge des données synthétiques et les retourne sous la forme d'un tableau numpy 
    (liste de listes pour faire simple).
    """
    return make_blobs(
        cluster_std=.15,
        centers=[[0, 1], 
                 [1, 0], 
                 [0, 0], 
                 [1, 1],
                 [.5, .5]],
        random_state=0
    )[0]

def charger_visages():
    """Charge les données Olivetti Faces et les retourne sous la forme d'un tableau numpy 
    (liste de listes pour faire simple).
    """
    return fetch_olivetti_faces().data

def visu_dendogramme(model):
    """Visualise le dendogramme correspondant à un modèle CAH fourni en entrée.

    Parameters
    ----------
    model : modèle scikit-learn
        Un modèle de CAH entraîné.
    """
    linkage_matrix = np.column_stack(
        [model.children_, 
         model.distances_, 
         np.zeros(model.children_.shape[0])]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix, 
        color_threshold=0. if model.n_clusters is None else model.distances_[::-1][model.n_clusters-2])
    plt.show()
    
def diagramme_en_batons_distances(modele, n_clusters_max=10):
    plt.bar(list(range(1, n_clusters_max + 1)), 
            modele.distances_[::-1][:n_clusters_max])
    plt.show()

def visu_donnees_synthetiques(X, clusters=None):
    """Visualise un jeu de données synthétique (généré par `charger_donnees`).

    Parameters
    ----------
    X : numpy.ndarray
        jeu de données (2D) à visualiser
    clusters : list ou `None`
        * si None : ne pas colorer les points
        * sinon   : clusters est une liste indiquant, pour chaque point 
          du jeu de données, le numéro du cluster auquel il est rattaché
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='k' if clusters is None else clusters)
    plt.show()
    
def visu_visages(X, clusters, h=64, w=64, n_visages_par_ligne=10):
    """Visualise un jeu de données Olivetti Faces (généré par :func:`charger_visages`).

    Parameters
    ----------
    X                   jeu de données de visages (images) à visualiser
    clusters            liste indiquant, pour chaque image du jeu de données, 
                        le numéro du cluster auquel elle est rattachée
    h                   hauteur des images (en pixels)
    w                   largeur des images (en pixels)
    n_visages_par_ligne nombre de visages à afficher pour chaque cluster
    
    """
    n_clusters = len(set(clusters))
    plt.figure()
    for i in range(n_clusters):
        for j in range(n_visages_par_ligne):
            plt.subplot(n_clusters, n_visages_par_ligne, i * n_visages_par_ligne + j + 1)
            plt.grid("off")
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel(f"Cluster {i + 1}")
            plt.imshow(X[clusters == i][j].reshape((h, w, 1)), cmap="Greys_r")
    plt.show()


