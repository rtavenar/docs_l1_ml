from sklearn.datasets import make_blobs, fetch_olivetti_faces
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

def charger_donnees():
    """Charge des données synthétiques et les retourne sous la forme d'un tableau numpy 
    (que vous manipulerez comme une liste de listes).

    Example
    -------
    >>> X = charger_donnees()
    >>> print(len(X))
    100
    >>> print(len(X[0]))
    2
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
    """Cette fonction renvoie un jeu de données représentant des images de visages en niveaux de gris.
    Chaque image, dans ce jeu de données, est représentée par la luminosité de chacun de ses pixels.
    Comme les images sont de taille 64x64, une image est représentée par un vecteur de taille 4096.

    Example
    -------
    >>> X = charger_visages()
    >>> print(len(X))
    400
    >>> print(len(X[0]))
    4096
    """
    return fetch_olivetti_faces().data

def visu_dendrogramme(model):
    """Visualise le dendrogramme correspondant à un modèle CAH fourni en entrée.

    Le dendrogramme permet de visualiser le process d’agglomération des individus.
    Il permet aussi de choisir le nombre de groupes car la hauteur des segments 
    correspond à la distance entre deux groupes (cf. p140 du CM).

    Parameters
    ----------
    model : modèle scikit-learn
        Un modèle de CAH entraîné pour lequel on a fixé ``compute_distances`` à ``True``.

    Example
    -------
    >>> cah = AgglomerativeClustering(compute_distances=True)
    >>> cah.fit(X)
    >>> visu_dendrogramme(cah)
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

def visu_donnees_synthetiques(X, clusters=None):
    """Visualise un jeu de données synthétique (généré par :func:`charger_donnees`).

    Si l'argument ``clusters`` est fourni, il doit contenir une liste d'entiers 
    de la même taille que X et chaque point du jeu de données sera coloré selon
    l'entier qui lui est associé.

    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (2D) à visualiser
    clusters : liste ou ``None``
        * si ``None`` : ne pas colorer les points
        * sinon   : ``clusters`` est une liste indiquant, pour chaque point 
          du jeu de données, le numéro du cluster auquel il est rattaché

    Examples
    --------
    >>> X = charger_donnees()
    >>> visu_donnees_synthetiques(X)
    >>> clusters = [1, 1, 7, 3, ..., 6]
    >>> visu_donnees_synthetiques(X, clusters)
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='k' if clusters is None else clusters)
    plt.show()

def diagramme_en_batons_distances(modele, n_clusters_max=10):
    """
    Cette fonction permet de représenter un diagramme en barres des distances entre groupes en fonction du nombre de groupes.
    On peut interpréter ce diagramme comme une mesure de la diﬀiculté à diminuer le nombre de groupes (cf. CM p141).

    Parameters
    ----------
    model : modèle scikit-learn
        Un modèle de CAH entraîné pour lequel on a fixé ``compute_distances`` à ``True``.
    n_clusters_max : ``int``
        Nombre maximum de clusters à inclure sur l'axe des abscisses.

    Examples
    --------
    Example
    -------
    >>> cah = AgglomerativeClustering(compute_distances=True)
    >>> cah.fit(X)
    >>> diagramme_en_batons_distances(cah, n_clusters_max=5)
    """
    plt.bar(list(range(1, n_clusters_max + 1)), 
            modele.distances_[::-1][:n_clusters_max])
    plt.show()
    
def visu_visages(X, clusters, h=64, w=64, n_visages_par_ligne=10):
    """Visualise un jeu de données Olivetti Faces (généré par :func:`charger_visages`).

    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données de visages (images) à visualiser
    clusters : liste
        liste indiquant, pour chaque image du jeu de données, 
        le numéro du cluster auquel elle est rattachée
    h : ``int``
        hauteur des images (en pixels)
    w : ``int``
        largeur des images (en pixels)
    n_visages_par_ligne : ``int``
        nombre de visages à afficher pour chaque cluster
    
    Example
    --------
    >>> X = charger_visages()
    >>> clusters = [1, 1, 7, 3, ..., 6]
    >>> visu_visages(X, clusters)
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


