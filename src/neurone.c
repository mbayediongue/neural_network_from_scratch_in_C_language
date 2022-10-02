/* 	Ce fichier contient les fonctions permettent la lecture de données depuis un fichier, de mettre ces données en entrée
* du réseau de neurones. Puis on fait la propagation et on compare les résultats en sortie avec ceux attendus */
#include <string.h>
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <math.h> /* Il faudra compiler en rajoutant : -lm pour inclure la biblio math */
#include "mlp.h"
#include "neurone.h"

/*Cette fonction fait l'allocation de mémoire pour un réseau
* tab est un tableau d'entiers contenant dans cet ordre: 
*          -le nombre de couches du réseau
*          -le nombre de neurones sur la couche d'entrée (couche 0)
*          -le nombre de neurone sur la couche 1
*          -...
*          -le nombre de neurones sur la couche de sortie
*/
t_perceptron * setPerceptron(t_perceptron *reseau,int tab[])
{
	int i,j,k;

	reseau->nombreCouches=tab[0];
	reseau->tabCouches=(t_couche *)malloc((reseau->nombreCouches)*sizeof(t_couche));
	
	for(i=0;i<reseau->nombreCouches;i++){
		(reseau->tabCouches)[i].nombreNeurones=tab[i+1];
		((reseau->tabCouches)[i]).tabNeurones=(t_neurone *)malloc(tab[i+1]*sizeof(t_neurone));
	}

	for(i=1;i<reseau->nombreCouches;i++){/* On commence par la couche 1 car les neurones de la couche 0 n'ont pas de poids*/

		for(j=0;j<tab[i+1];j++){	/* tab[i+1] est le nombre de neurones de la couche i */
			/* Allocation de mémoire pour les poids reliant le neurone j de la couche i aux neurones de la couche i-1 qui sont au nombre de tab[i]*/
			(((reseau->tabCouches)[i]).tabNeurones)[j].w=(double *)malloc(tab[i]*sizeof(double));
			for(k=0;k<tab[i];k++){
				reseau->tabCouches[i].tabNeurones[j].w[k]=-1+2*(rand()/(1.0+RAND_MAX)); /*  un nombre aléatoire en -1 et 1*/
			}
		}
	}
	return reseau;
}


/* Cette fonction permet de construire le réseau de neurone à partir d'un fichier contenant les poids entre les différents neurones */
t_perceptron * perceptronLecture(char *fichier,t_perceptron *reseau)
{
	FILE *f;
	int nb_couches,*tab;/* tab sera un tableau d'entiers (voire la description au-dessus de la fonction setPerceptron*/
	int k,i,j,l;
	double w;
	char info[100];

	f=fopen(fichier,"r");
	if(f==NULL) {
		printf("Impossible de lire le fichier <%s>",fichier);
		return NULL;
	}

	/* On le lit nombre de lignes sur la première couche*/
	fscanf(f,"%d",&nb_couches);
	printf("\n Nombre de couches_: %d", nb_couches);
	tab=(int *)malloc(sizeof(int)*nb_couches);

	tab[0]=nb_couches;
	
	for(i=0;i<nb_couches;i++){
		/* On 'saute' le message " : Nombre de neurone de la couche 2" (exactement 8 mots)*/
		while(fgetc(f)!='\n');
		
		/* On est arrivé en fin de ligne, la prochaine info lue sera le nombre de neurones de la ligne suivante*/
		fscanf(f,"%d",&tab[i+1]);
		printf("\n Nombre de neurones couche %d : %d", i, tab[i]);

	}

	setPerceptron(reseau,tab);/* allocation de mémoire pour le réseau */

	/* On 'saute' le message " : Nombre de neurone de la couche 2" (exactement 8 mots)*/
	for(l=0;l<8;l++){
		fscanf(f,"%s",info);
	}
	
	for(i=1;i<nb_couches;i++){/* On commence par la couche 1 car les neurones de la couche 0 n'ont pas de poids*/
		
		for(j=0;j<tab[i+1];j++){	/* tab[i+1] est le nombre de neurones de la couche i*/
			/* Puis on saute le message: "Poids du neurone j de la couche i" (contenant exactement 8 chaines de caractères)*/
			for(l=0;l<8;l++) fscanf(f,"%s",info);
			
			for(k=0;k<tab[i];k++){
				fscanf(f,"%lf",&w);
				(((reseau->tabCouches)[i]).tabNeurones)[j].w[k]=w;
			}
		}
	}
	fclose(f);
	return reseau;
}


/* Cette fonction prend en entrée une couche et des poids correspondant chacun à un neurone de la couche*
* Elle retourne la fonction signoide correspondante */
double sigmoide(t_couche couche,double weight[])
{
	int i;
	double s=0;

	for(i=0;i<couche.nombreNeurones;i++){
		s=s+weight[i]*(couche.tabNeurones[i].x);
	}	
	return 1/(1+exp(-s));
}


int perceptronPropagation(t_perceptron *reseau)
{
	int i,j,indice_max,nb_couches; 
	double sortie_max;

	nb_couches=reseau->nombreCouches;
	/* On remplit les valeurs de sorties de la couches 1 à celle de sortie
	* On suppose dans cette fonction que les neurones de la couche 0 ont pour valeur de sortie 
	* les données renseignées en entrée du réseau*/
	for(i=1;i<nb_couches;i++){

		for(j=0;j< reseau->tabCouches[i].nombreNeurones ;j++){
			(reseau->tabCouches[i]).tabNeurones[j].x=sigmoide(reseau->tabCouches[i-1],(reseau->tabCouches[i]).tabNeurones[j].w);
		}
	}

	/* On cherche ensuite le maximum entre les valeurs de sortie de la dernière couche
	Puis on retourne l'indice correspondant */
	sortie_max=((reseau->tabCouches[nb_couches-1]).tabNeurones[0]).x ;
	indice_max=0;

	for(i=1;i<(reseau->tabCouches[nb_couches-1]).nombreNeurones;i++){
		
		if(((reseau->tabCouches[nb_couches-1]).tabNeurones[i]).x > sortie_max){
			sortie_max=((reseau->tabCouches[nb_couches-1]).tabNeurones[i]).x;
			indice_max=i;
		}
	}	
	return indice_max;
}


void perceptronTest(t_perceptron *reseau,char *fichier)
{
	FILE *f;
	int i,j, c,nb_erreur=0,nb_entree,nb_sortie,nb_donnees=0,resultat,indice_max;
	double max,Erreur_globale=0,resultat_attendu;

	f=fopen(fichier,"r");
	if(f==NULL){
		printf(" Impossible d'ouvrir le fichier %s \n",fichier );
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

	for(i=0;i<nb_donnees;i++){

		/* On remplit la sortie des neurones de la couche 0 par les données se trouvant dans le fichier*/
		for(j=0;j<nb_entree;j++){

			fscanf(f,"%lf",&(reseau->tabCouches[0].tabNeurones[j].x));
		}

		/* Il faut enseuite propager l'info dans les couches successives du réseau de neurone*/
		resultat=perceptronPropagation(reseau);

		/* On cherche le maximum attendu dans la couche de sortie selon les données du fichier */
		fscanf(f,"%lf",&max);
		indice_max=0;
		for(j=1;j<nb_sortie;j++){
			fscanf(f,"%lf",&resultat_attendu);

			if(resultat_attendu > max){
				max=resultat_attendu;
				indice_max=j;
			}

		}

		if(indice_max != resultat){/*Le résulat en sortie du neurone n'est pas celui attendu*/
			/*printf("\n\n=> Le traitement par le réseau de neurones de l'exemple %d ne donne pas le résultat attendu (sortie attendue: %d)\n",i,indice_max);
			
			printf("Voici la couche de sortie du réseau:\n");
			*/
			for(j=0;j<nb_sortie;j++) 
				/*printf(" %f \t",reseau->tabCouches[(reseau->nombreCouches)-1].tabNeurones[j].x);*/

			Erreur_globale+=pow((indice_max-resultat),2);
			nb_erreur++;
		}

	}
	fclose(f);
	/*deleteReseau(reseau);*/
	printf("\n\n *L'erreur globale est %f et le nombre total d'erreurs est de %d sur %d *\n \n",Erreur_globale,nb_erreur,nb_donnees);

}

/*Cette fonction libère  la moi alloué au réseau de neurones et le supprime ainsi */
void deleteReseau(t_perceptron *reseau)
{
	int i, j;

	for(i=0;i<reseau->nombreCouches;i++){
		for(j=0;j<reseau->tabCouches[i].nombreNeurones;j++){
			if(i>0)
				free(reseau->tabCouches[i].tabNeurones[j].w);
		}
		free(reseau->tabCouches[i].tabNeurones);
	}
	free(reseau->tabCouches);
}