import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import os
from math import ceil, sqrt
from matplotlib import image as mpimg
from pandas import DataFrame

def load_data_citycrime():
    """Charge le jeu de données City Crime.
    
    Returns
    -------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    villes : liste de chaînes de caractères
        noms des villes concernées
    y : liste de labels
        variable cible (=classe)

    Example
    -------
    >>> X, villes, y = load_data_citycrime()
    """
    with open('citycrime.dat') as f:
        lines = f.readlines()

    data = []
    villes = []
    for i, line in enumerate(lines):
        if i != 0:
            data_obs = []
            for elmt in line.replace('\n', '').split(' '):
                if elmt != '':
                    try:
                        data_obs.append(float(elmt))
                    except:
                        villes.append(elmt)
            data.append(data_obs)
        else:
            crimes = []
            for elmt in line.replace('\n', '').split(' '):
                if elmt != '':
                    crimes.append(elmt)
    return data, villes, crimes

def visu_dendrogramme(modele, villes=[]):
    """Visualise le dendrogramme correspondant à un modèle CAH fourni en entrée.

    Le dendrogramme permet de visualiser le process d’agglomération des individus.
    Il permet aussi de choisir le nombre de groupes car la hauteur des segments 
    correspond à la distance entre deux groupes.

    Parameters
    ----------
    modele : modèle scikit-learn
        Un modèle de CAH entraîné pour lequel on a fixé ``compute_distances`` à ``True``.
    villes : liste
        Une liste des noms de villes à afficher

    Example
    -------
    >>> cah = AgglomerativeClustering(compute_distances=True)
    >>> cah.fit(X)
    >>> visu_dendrogramme(cah)
    >>> visu_dendrogramme(cah, ["Marseille", "Montpellier", ...])
    """
    linkage_matrix = np.column_stack(
        [modele.children_,
         modele.distances_,
         np.zeros(modele.children_.shape[0])]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(
        linkage_matrix, labels=villes, orientation="left",
        color_threshold=0. if modele.n_clusters is None else modele.distances_[::-1][modele.n_clusters - 2])
    plt.show()

def diagramme_en_batons_distances(modele, n_clusters_max=10):
    """Trace un diagramme en barres des distances entre groupes en fonction du nombre de groupes.
    On peut interpréter ce diagramme comme une mesure de la difficulté à diminuer le nombre de groupes.

    Parameters
    ----------
    modele : modèle scikit-learn
        Un modèle de CAH entraîné pour lequel on a fixé ``compute_distances`` à ``True``.
    n_clusters_max : ``int``
        Nombre maximum de clusters à inclure sur l'axe des abscisses.

    Example
    -------
    >>> cah = AgglomerativeClustering(compute_distances=True)
    >>> cah.fit(X)
    >>> diagramme_en_batons_distances(cah, n_clusters_max=5)
    """
    plt.bar(list(range(1, n_clusters_max + 1)),
            modele.distances_[::-1][:n_clusters_max])
    plt.show()

def charger_paysages():
    """Charge le jeu de données de classification d'images de paysage.
    
    Ce jeu de données contient des images de paysage dans cinq catégories : côte, désert, forêt, glacier et montagne.
    Chaque image est caractérisée par un vecteur de dimension 512 représentant son histogramme de couleurs avec 8x8x8=512 bins.
    Chaque image est associée à un label parmi 'cote', 'desert', 'foret', 'glacier', 'montagne'.

    Returns
    -------
    X : ``numpy.ndarray`` (liste de liste)
        jeu de données (variables explicatives)
    y : liste de labels
        variable cible (=classe)

    Example
    -------
    >>> X, y = charger_paysages()
    """
    loaded = np.load('paysages.npz')
    X = loaded["X"]
    y = DataFrame(loaded["y"], columns=['label'])
    return X, y


def liste_fichiers_dossier(dossier):
    """Renvoie la liste des chemins des fichiers contenus dans le dossier spécifié.

    Parameters
    ----------
    dossier : chaîne de caractères
        Nom du dossier à explorer
    
    Returns
    -------
    liste_fichiers : liste de chaînes de caractères
        Liste des chemins des fichiers trouvés dans le dossier spécifié

    Example
    -------
    >>> liste_fichiers_images = liste_fichiers_dossier("paysages"")
    """
    liste_fichiers = []
    for (repertoire, sousRepertoires, fichiers) in os.walk(dossier):
        liste_fichiers.extend([os.path.join(repertoire, fichier) for fichier in fichiers])
    liste_fichiers = sorted(liste_fichiers)
    return liste_fichiers


def afficher_images_label(liste_chemins_images, label, y_test=None, y_pred=None, n=30):
    """Affiche un échantillon d'image pour un label donné.

    Si ni ``y_test`` ni ``y_pred`` n'est fourni, l'échantillon d'images pour le label spécifié
    est pris dans le jeu de données complet.
    Si ``y_test`` est fourni (sans ``y_pred``) l'échantillon d'images pour le label spécifié est
    pris dans le jeu de données de test.
    Si ``y_test`` et ``y_pred`` sont tous les deux fournis l'échantillons d'image est pris dans
    les images prédites pour le label spécifié.

    Parameters
    ----------
    liste_chemins_images : liste de chaînes de caractères
        Liste des chemins des images du jeu de données complet
    label : liste
        Label des images à afficher
    y_test : liste (facultatif)
        Liste des labels des images du jeu données de test
    y_pred : liste (facultatif) 
        Liste des labels prédits pour les images du jeu données de test 
        (ne peut être utilisé sans fournir ``y_test``)
    n : ``int`` (facultatif, valeur par défaut : 30)
        Nombre d'images à afficher dans l'échantillon

    Example
    -------
    >>> afficher_images_label(liste_chemins_images, "foret")
    >>> afficher_images_label(liste_chemins_images, "cote", y_test=my_y_test)
    >>> afficher_images_label(liste_chemins_images, "glacier", y_test=my_y_test, y_pred=my_y_pred)
    """
    plt.figure(figsize=(12,9))
    if y_test is None and y_pred is None:
        plt.suptitle(f'Echantillon d\'images du jeu de données complet pour la classe {label}')
        indices = [i for i, chemin in enumerate(liste_chemins_images) if label in chemin]
    elif y_test is not None and y_pred is None:
        plt.suptitle(f'Echantillon d\'images du jeu de données de test pour la classe {label}')
        indices = [i for i in y_test.index if label in liste_chemins_images[i]]
    elif y_test is not None and y_pred is not None:
        plt.suptitle(f'Echantillon d\'images prédites pour la classe {label}')
        indices = [y_test.index[i] for i in range(len(y_pred)) if y_pred[i]==label]
    else :
        plt.suptitle(f'Pour afficher des images prédites pour la classe {label} vous devez fournir les parmètres y_pred ET y_test')
        plt.show()
        return None

    n = min(len(indices),n)
    np.random.seed(1)
    indices = np.random.choice(indices, size=n, replace=False)

    for i, ind in enumerate(indices):
        plt.subplot(ceil(sqrt(5*n/6)),ceil(sqrt(6*n/5)),i+1)
        chemin_image = liste_chemins_images[ind]
        image = mpimg.imread(chemin_image)
        plt.title(os.path.splitext(os.path.basename(chemin_image))[0].replace("-Train", ""), fontsize = 8, y=1.0, pad=-8)
        plt.imshow(image)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0.01) 
    plt.show()


def afficher_matrice_confusion(y_test, y_pred, labels):
    """Affiche la matrice de confusion pour une liste de labels de test, une liste de
    labels prédits et une liste des labels possibles

    Parameters
    ----------
    y_test : liste
        Liste des labels du jeu de données de test
    y_pred : liste
        Liste des labels prédits pour le jeu de données de test
    labels : liste
        Liste de labels possibles

    Example
    -------
    >>> afficher_matrice_confusion(my_y_test, my_y_pred, my_labels):
    """
    print("Matrice de Confusion :")
    from sklearn.metrics import confusion_matrix
    print(len(y_test), len(y_pred))
    matrice_confusion = confusion_matrix(y_test, y_pred, labels=labels)
    print(matrice_confusion)
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    df_cm = pd.DataFrame(matrice_confusion, index=labels,
                         columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


