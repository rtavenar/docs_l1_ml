from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
import graphviz

def charger_mnist():
    """Charge le jeu de données de classification d'images MNIST.
    
    Ce jeu de données contient des images de chiffres manuscrit et la tâche de classification
    consiste à retrouver le chiffre écrit dans l'image (10 classes possibles, de 0 à 9).
    Chaque image est en résolution 28x28, ce qui fait 784 pixels en tout.

    Returns
    -------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    y : liste d'entiers
        variable cible (=classe)

    Example
    -------
    >>> X, y = charger_mnist()
    >>> print(len(X))
    60000
    >>> print(len(y))
    60000
    >>> print(len(X[0]))
    784
    """
    X, y = mnist.load_data()[0]
    X = X.reshape((X.shape[0], -1))
    return X, y


def visu_mnist(X, y, preds=None):
    """Visualise un jeu de données d'images de résolution 28x28

    Parameters
    ----------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    y : liste d'entiers
        variable cible (=classe)
    preds : liste d'entiers, ou ``None``
        liste des prédictions fournies par un modèle. 
        Si ``None``, on ne visualise pas les informations 
        liées aux prédictions.

    Example
    -------
    >>> X, y = charger_mnist()
    >>> visu_mnist(X, y)
    >>> visu_mnist(X_test, y_test, modele.predict(X_test))
    """
    np.random.seed(0)
    plt.figure(figsize=(8, 8))
    indices = np.random.choice(len(X), size=16, replace=False)
    for i, idx in enumerate(indices):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(X[idx].reshape((28, 28)), cmap="Greys_r")
        if preds is None:
            plt.title(f"Image {idx}\nClasse {y[idx]}")
        else:
            plt.title(f"Image {idx}\nClasse {y[idx]}, Prédite {preds[idx]}")
            if preds[idx] != y[idx]:
                plt.setp(ax.spines.values(), color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()

def visu_arbre(modele_arbre):
    """Visualise un jeu de données d'images de résolution 28x28

    Parameters
    ----------
    modele_arbre : modèle scikit-learn de type ``DecisionTreeClassifier``
        Modèle à visualiser. 
        Ce modèle doit avoir été ajusté (=entraîné) sur un jeu de 
        données avant de pouvoir être visualisé.

    Example
    -------
    >>> m = DecisionTreeClassifier(...)
    >>> m.fit(X, y)
    >>> visu_arbre(m)
    """
    dot_data = export_graphviz(modele_arbre, out_file=None, filled=True, rounded=True)
    return graphviz.Source(dot_data)

def feature_importance_viz(modele_arbre):
    """Visualise les pixels sur lesquels un modèle de type arbre de 
    décision base ses décisions.
    Suppose que le modèle a été entraîné sur un jeu de données 
    d'images de résolution 28x28.

    Parameters
    ----------
    modele_arbre : modèle scikit-learn de type ``DecisionTreeClassifier``
        Modèle pour lequel générer la visualisation. 
        Ce modèle doit avoir été ajusté (=entraîné) sur un jeu de 
        données avant de pouvoir être visualisé.

    Example
    -------
    >>> m = DecisionTreeClassifier(...)
    >>> m.fit(X, y)
    >>> feature_importance_viz(m)
    """
    imp = modele_arbre.feature_importances_
    plt.imshow(imp.reshape((28, 28)), cmap="Greys")
    plt.xticks([])
    plt.yticks([])