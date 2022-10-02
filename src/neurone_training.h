/* Ce fichier contient les entetes des fonctions dans neurone.training.c*/

void EntrainementReseau(t_perceptron *,char *,char *, int);
void perceptronAjustePoids(t_perceptron *,double **,double);
void perceptronRetroPropagation(t_perceptron *,double **);
void perceptronEntrainementUnExemple(t_perceptron *,double *,double **,double);
double perceptronErreurGlobale(t_perceptron *, double *);
int perceptronSave(t_perceptron *,char *);
void perceptronErreur(t_perceptron *, double *,double *);
