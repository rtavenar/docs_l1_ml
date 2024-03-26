def diagramme_en_batons(valeurs_en_x, valeurs_en_y):
    """Cette fonction affiche un diagramme en batons à partir
    des listes des coordonnées en x (valeurs_en_x) et des
    coordonnées en y (valeurs_en_y).
    
    Parameters
    ----------
    valeurs_en_x: liste
        Liste des étiquettes pour l'axe des abscisses
    valeurs_en_y: liste
        Liste des hauteurs de bâtons (axe des ordonnées)

    Example
    -------
    >>> diagramme_en_batons([1, 2, 3], [10, 7, 3])
    """
    plt.figure()
    rects = plt.bar([f"k={xi}" for xi in valeurs_en_x], valeurs_en_y)
    plt.bar_label(rects)
    plt.show()