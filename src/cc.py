import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def charge_donnees():
    """Charge le jeu de données et le divise en ensembles d'apprentissage et de test.
    
    Returns
    -------
    X_train : numpy.ndarray ("liste de liste")
        Données d'apprentissage (variables explicatives)
    X_test : numpy.ndarray ("liste de liste")
        Données de test (variables explicatives)
    y_train : numpy.ndarray ("liste")
        Labels d'apprentissage
    y_test : numpy.ndarray ("liste")
        Labels de test
    
    Example
    -------
    >>> X_train, X_test, y_train, y_test = charge_donnees()
    """
    X, y = make_moons(n_samples=1000, noise=0.15, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)

def visu_donnees(X, y):
    """Visualise les données avec un nuage de points coloré par classe.
    
    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    y : liste de labels
        variable cible (=classe)
    
    Example
    -------
    >>> visu_donnees(X, y)
    """
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Classe 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Classe 1")
    plt.legend()
    plt.show()

def visu_clusters(X, model):
    """Visualise les clusters obtenus par un modèle de clustering.
    
    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    model : sklearn.cluster model
        Modèle de clustering entraîné
    
    Example
    -------
    >>> visu_clusters(X, model)
    """
    model.fit(X)
    clusters = model.labels_
    for i in range(model.n_clusters_):
        plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f"Cluster {i}")
    plt.legend()
    plt.show()

def diagramme_en_batons_distances(modele, n_clusters_max=10):
    """Affiche un diagramme en bâtons des distances pour déterminer le nombre optimal de clusters.
    
    Parameters
    ----------
    modele : sklearn.cluster.AgglomerativeClustering
        Modèle de clustering hiérarchique avec compute_distances=True
    n_clusters_max : int, default=10
        Nombre maximum de clusters à afficher
    
    Example
    -------
    >>> diagramme_en_batons_distances(model)
    """
    plt.bar(list(range(1, n_clusters_max + 1)), 
            modele.distances_[::-1][:n_clusters_max])
    plt.show()

def visu_frontiere(X_test, y_test, model):
    """Affiche la frontière de décision d'un modèle de classification en 2D.
    
    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    y : liste de labels
        variable cible (=classe)
    model : sklearn model
        Modèle de classification entraîné
    
    Example
    -------
    >>> visu_frontiere(X_test, y_test, model)
    """
    x_min, x_max = X_test[:, 0].min() - .1, X_test[:, 0].max() + .1
    y_min, y_max = X_test[:, 1].min() - .1, X_test[:, 1].max() + .1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k')

    plt.title("Frontière de décision")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
