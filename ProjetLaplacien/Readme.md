# Description du projet

Le but du projet est de mettre en oeuvre différentes méthodes de décomposition de domaine
pour résoudre un problème de laplacien sur un carré triangulé.

Dans un premier temps, nous allons utiliser une méthode de gradient conjugué et paralléliser le produit matrice-vecteur
selon le découpage soit (au choix) :

- par élément
- par noeud

Vous aurez le choix entre :

1. Utiliser une bissection emboîtée en itérant une décomposition de domaine
2. Utiliser une méthode FETI 1-LM
3. Utiliser une méthode FETI 2-LM

On comparera du point de vue performance la seconde méthode choisie par rapport à la première méthode sélectionnée.

# Présentation des fichiers :

Vous trouverez dans le projet :

- Trois maillages de carré (petit, moyen et gros) : CarrePetit.msh, CarreMedium.msh et CarreGros.msh
- Un fichier décrivant la géométrie d'un carré pour regénérer des maillages de carré avec gmsh : Carre.geo
- Un fichier conjugate_gradient.py permettant d'utiliser un gradient conjugué
- Le fichier fem_laplacian permet d'assembler des matrices élémentaires pour le laplacien
- Le fichier fem.py permet quant à lui de construire le squelette de la matrice creuse et rajouter des matrices élémentaires à la matrice creuse
- Le fichier hilbert-curve sert pour le découpeur
- Le fichier splitter.py est un module permettant de découper un maillage par noeud ou par élément. Un exemple d'utilisation est donné à la fin du script
- Le fichier visu_solution.py permet de visualiser la solution
- Le fichier visu_split_mesh.py permet de visualiser le découpage obtenu
- Le fichier mesh.py permet de lire un maillage et de calculer sa connectivité
- SparseMatrix_direct.py est un solveur permettant de résoudre à l'aide d'un solveur direct le problème du laplacien en séquentiel. Ce fichier sert comme base pour la bissection emboîtée et les méthodes FETIs
- SparseMatrix_iterative.py est un solveur utilisant un gradient conjugué . Ce fichier sert comme base pour le gradient conjugué parallélisé et les méthodes FETI.
- Les deux fichiers pdf permettent de faire la parallélisation du produit matrice-vecteur.