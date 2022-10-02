#ifndef _NEURONE_H_
#define _NEURONE_H_
/* Ce fichier contient les entetes des fonctions dans neurone.c
* Ces fonctions permettent la lecture de données depuis un fichier, de mettre ces données en entrée
* du réseau de neurones; Puis on fait la propagation et on compare les résultats en sortie avec ceux attendus*/

t_perceptron * perceptronLecture(char *,t_perceptron *);
t_perceptron * setPerceptron(t_perceptron *,int *);
t_perceptron * initializeNetwork(int *, t_perceptron *);
double sigmoide(t_couche ,double *);
int perceptronPropagation(t_perceptron *);
void perceptronTest(t_perceptron *,char *);
void deleteReseau(t_perceptron *);

#endif