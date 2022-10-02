/* Ce fichier contient les fonctions permettant d'entrainer un réseau de neurones (i.e définir les poids entre les neurones)
* à partir d'exemples */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"
#include "math.h"
#include "time.h"
#include "neurone.h"
#include "neurone_training.h"
#define PAS 0.5


void EntrainementReseau(t_perceptron *reseau,char *fichier_test,char *nomReseauEntraine, int max_iter)
{
	FILE *f;
	int i,j,c,nb_entree,nb_sortie,nb_donnees=0,nb_iterations=1;
	double *cible,**TabErreur,ErreurTotale,pas;
	clock_t before=clock(),difference;
	int duration;

	f=fopen(fichier_test,"r");
	if(f==NULL){
		printf(" Impossible d'ouvrir le fichier test %s \n",fichier_test );
		return;
	}

	/* Calculons le nombre de données disponibles (i.e le nombre de lignes ici)*/
	while((c=fgetc(f))!=EOF){
		if(c=='\n') nb_donnees++;
	}
	
	nb_donnees-=2; /* On enlève 2 car les 2 premières lignes ne sont pas des données sur les fleurs (elles contiennent le nombre de neurone dans
					* la couche d'entrée et celui de la couche de sortie) */
	
	printf("Le nombre de donneées est %d\n", nb_donnees);
	
	/* On revient en début de fichier */
	rewind(f);

	/* Enregistrons le nombre de neurones de la couche d'entrées et celui dans la couche de sortie (ces infos sont sur la 1ère et 
	 * la 2e ligne resoectivement) */
	fscanf(f,"%d",&nb_entree);
	while(fgetc(f)!='\n');
	fscanf(f,"%d",&nb_sortie);
	while(fgetc(f)!='\n');
	
	printf("Nb sortie %d", nb_sortie);
	cible=(double *)malloc(sizeof(double)*nb_sortie);
	if(cible==NULL){
		printf("Allocation de mémoire pour la cible impossible\n");
		exit(1);
	}

	TabErreur=(double **)malloc(sizeof(double *)*(reseau->nombreCouches));
	if(TabErreur==NULL){
		printf("Allocation de mémoire tabErreur impossible\n");
		exit(1);
	}
	for(i=0;i<reseau->nombreCouches;i++){
		TabErreur[i]=(double *)malloc(sizeof(double)*(reseau->tabCouches[i].nombreNeurones));		
		if(TabErreur[i]==NULL){
			printf("Allocation de mémoire des champs de tabErreur impossible\n");
			exit(1);
		}	
	}

	pas=PAS;
	/* Puis on commence l'entrainement ;) */
	do{
		ErreurTotale=0;
		
		printf("Le pas est de %f\n",pas);
		for(i=0;i<nb_donnees;i++){
			/* On remplit la sortie des neurones de la couche 0 par les données se trouvant dans le fichier*/
			for(j=0;j<nb_entree;j++){
				fscanf(f,"%lf",&(reseau->tabCouches[0].tabNeurones[j].x));
			}

			for(j=0;j<nb_sortie;j++){
				fscanf(f,"%lf",&cible[j]);
			}

			perceptronEntrainementUnExemple(reseau,cible,TabErreur,pas);
			ErreurTotale+=perceptronErreurGlobale(reseau,cible);
		}
		/* On revient en début de fichier et on saute les 2 premières lignes...*/
		rewind(f);
		fscanf(f,"%d",&nb_entree);
		while(fgetc(f)!='\n');
		fscanf(f,"%d",&nb_sortie);
		while(fgetc(f)!='\n');

		nb_iterations++;

		/* On ajuste le pas...*/
		if(nb_iterations >= 3){
			pas=PAS/(log(nb_iterations));
		}

		difference=clock()-before;
		duration=difference/CLOCKS_PER_SEC;
		printf("Apprentissage en cours... %dème itération,temps écoulé: %dmin%ds \n", nb_iterations+1,duration/60,duration%60);
		
	
	}while((ErreurTotale>=(nb_sortie/2)*(nb_donnees/10))&&(nb_iterations<max_iter)&&(duration<60*60*5)); /* On recommence l'opération jusqu'à que le nombre d'erreurs soient 
												* soit "petite" ou qu'on ait fait un maximum fixé  itérations*/
	printf(" \n  * %d itérations ont été faites pour l'apprentissage *\n",nb_iterations );
	
	fclose(f);
	perceptronSave(reseau,nomReseauEntraine);

	for(i=0;i<reseau->nombreCouches;i++){
		free(TabErreur[i]);
	}

	free(TabErreur);
	TabErreur=NULL;

	printf("\n *L'entrainement du réseau de neurones avec les exemples du fichier %s est terminé\n", fichier_test);

}


/* reseau est un réseau de neurones dont les entrées sont déjà renseignées
* Cible est un tableau contenant les valeurs de sortie attendue */
void perceptronEntrainementUnExemple(t_perceptron *reseau,double *cible,double **TabErreur,double pas)
{

	perceptronPropagation(reseau);

	perceptronErreur(reseau,cible,TabErreur[(reseau->nombreCouches)-1]);

	perceptronRetroPropagation(reseau,TabErreur);

	perceptronAjustePoids(reseau,TabErreur,pas);

}


/* Cette fonction calcule l'erreur de CHAQUE neurone de la couche de sortie comparée à celle attendue*/

void perceptronErreur(t_perceptron *reseau, double *cible,double *erreurSortie)
{
	double x;
	int i,nbCouches;

	nbCouches=reseau->nombreCouches;

	for(i=0; i < reseau->tabCouches[nbCouches-1].nombreNeurones;i++){
		x=reseau->tabCouches[nbCouches-1].tabNeurones[i].x;

		erreurSortie[i]=x*(1-x)*(cible[i]-x);
	}
}

/* Cette fonction calcule l'erreur des couches cachées par rétropropagation du gradient */

void perceptronRetroPropagation(t_perceptron *reseau,double **TabErreur)
{
	double x,s; /* TabErreur contiendra l'erreur de toutes les couches */
	int i,j,k,nbCouches;

	nbCouches=reseau->nombreCouches;

	for(i=nbCouches-2;i>=0;i--){/* on descend de l'avant dernière couche jusqu'à la première */

		for(j=0; j < reseau->tabCouches[i].nombreNeurones;j++){/*Pour chaque neurone de la couche cachée i calculons l'erreur...*/
			
			x=reseau->tabCouches[i].tabNeurones[j].x;

			s=0;
			for(k=0;k<reseau->tabCouches[i+1].nombreNeurones; k++){

				s+=(TabErreur[i+1][k])*(reseau->tabCouches[i+1].tabNeurones[k].w[j]);
			}

			TabErreur[i][j]=x*(1-x)*s;
			/*printf(" Erreur neurone %d couche %d : %f  ",j,i,TabErreur[i][j]);
			*/
		}
	}
}
/* Cette fonction fait l'ajustment des poids à partir des erreurs calculées*/
void perceptronAjustePoids(t_perceptron *reseau,double **TabErreur,double pas)
{
	int i,j,k;
	double x;

	for(i=1;i<reseau->nombreCouches;i++){
		for(j=0;j<reseau->tabCouches[i].nombreNeurones;j++){

			for(k=0;k<reseau->tabCouches[i-1].nombreNeurones;k++){
				x=reseau->tabCouches[i].tabNeurones[j].x;
				(reseau->tabCouches[i].tabNeurones[j].w[k])+=pas*(TabErreur[i][j])*(1-x);
			}
		}
	}
}


/* Cette fonction sauvegarde le réseau issu de l'apprentissage dans un fichier*/
int perceptronSave(t_perceptron *reseau,char *nomReseauEntraine)
{
	FILE *f;
	int i,j,k;
	double sz;

	f=fopen(nomReseauEntraine,"w");
	if(f==NULL){
		printf("Création de fichier de sauvegarde du réseau entrainé impossible \n");
		return -1;
	}
	fprintf(f,"%d : Nombre de couches\n",reseau->nombreCouches);

	for(i=0;i<reseau->nombreCouches;i++){

		fprintf(f,"%d : Nombre de neurones de la couche %d\n",reseau->tabCouches[i].nombreNeurones,i);
	}

	for(i=1;i<reseau->nombreCouches;i++){

		for(j=0;j<reseau->tabCouches[i].nombreNeurones;j++){
			fprintf(f, "Poids du neurone %d de la couche %d\n",j,i);
		
			sz=reseau->tabCouches[i-1].nombreNeurones;
			for(k=0;k<sz;k++){
			/*fwrite(reseau->tabCouches[i].tabNeurones[j].w,sizeof(double),sz,f);*/
				fprintf(f,"%f  ",reseau->tabCouches[i].tabNeurones[j].w[k]);
			}
			fprintf(f,"\n");
		}
	}
	fclose(f);
	return 1;
}

/*Cette fonction calcule l'erreur TOTALE de sortie par rapport à celle sésirée */
double perceptronErreurGlobale(t_perceptron *reseau, double *cible)
{
	double erreur=0;
	int i,nbCouches;
	nbCouches=reseau->nombreCouches;

	for(i=0;i<reseau->tabCouches[nbCouches-1].nombreNeurones;i++){
		erreur+=pow(cible[i]-(reseau->tabCouches[nbCouches-1].tabNeurones[i].x) , 2);
	}
	return erreur;
}