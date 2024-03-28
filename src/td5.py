from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz
import graphviz

def prepare_mnist():
    X, y = mnist.load_data()[0]
    X = X.reshape((X.shape[0], -1))
    return X, y


def dataviz(X, y, preds=None):
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

def visu_arbre(model_arbre):
    dot_data = export_graphviz(model_arbre, out_file=None, filled=True, rounded=True)
    graphviz.Source(dot_data)

def feature_importance_viz(model_arbre):
    imp = model_arbre.feature_importances_
    plt.imshow(imp.reshape((28, 28)), cmap="Greys")
    plt.xticks([])
    plt.yticks([])