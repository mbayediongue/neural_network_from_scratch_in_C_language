#ifndef _MLP_H_
#define _MLP_H_


/* Le type definissant un neurone */
typedef struct Neurone {
double  x;      /* la valeur de sortie du neurone */
double* w;     /* tableau des poids des connexions entre ce neurone et ceux de la couche inferieur */
} t_neurone;

/* Le type definissant une couche */
typedef struct Couche {
int     nombreNeurones;    /* Nombre de neurones de la couche */
t_neurone* tabNeurones;  /* Tableau des neurones de la couche */
} t_couche;

/* Le type definissant un reseau */
typedef struct Perceptron {
int    nombreCouches;    /* Nombre de couches du reseau, en comptant les couches d'entree et de sortie */
t_couche* tabCouches;    /* Tableau des couches de neurones */
} t_perceptron;



#endif
