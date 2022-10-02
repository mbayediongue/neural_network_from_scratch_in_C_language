/* 
*  Implement a feed-forward neural network from scratch in C.	
*
* Authot: Mbaye DIONGUE
* Date: November 2020
*
*/

#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"
#include "neurone.h"
#include "neurone_training.h"


int main(int argc, char *argv[])
{
	t_perceptron reseau1;

	/* Variable tab: Contient le nombre de couches et le nombre de neurones par couche
	* tab[0] = nombre de couches du réseau
	* tab[1] = nombre de neurones de la 1ère couche
	* tab[2] = nombre de neurones de la 2ème couche
	* ...
	* ect.
	*/
	int *tab; 
	int nb_couches=3,max_iter=5; /* paramètre pouvant être modifiés par l'utilisateur */    
	int end_of_loop=0;
	char nom_reseau[50],exemples_entrainement_test[50],choix;

	tab=(int *)malloc(sizeof(int)*(nb_couches));
	tab[0]=nb_couches;
	tab[1]=80;
	tab[2]=10;
	
	printf("Appuyez sur 't' pour Tester un réseau de neurone déjà entrainé\n"
	/*"\nAppuyez sur 'e' pour Entrainer un réseau de neurone\n"*/
	"Appuyer sur 'f' pour Fermer le programme\n");
	
	do{
		scanf("%c",&choix);
		switch(choix){
			case 'e':
				printf("Veuillez saisir le nom du fichier contenant les exemples pour l'entrainment :\n");
				scanf("%s",exemples_entrainement_test);
				printf("Saisissez maintenant le nom de fichier où vous voulez enregistrer le résultat de l'entrainement:\n");
				scanf("%s",nom_reseau);

				setPerceptron(&reseau1,tab);/* allocation de mémoire pour le réseau */
				EntrainementReseau(&reseau1,exemples_entrainement_test,nom_reseau,max_iter);

				break;

			case 't':
				printf("Veuillez saisir le nom du réseau de neurones que vous voulez tester :\n");
				scanf("%s",nom_reseau);
				perceptronLecture(nom_reseau,&reseau1);

				printf("\nSaisissez maintenant le nom du fichier de test:\n");
				scanf("%s",exemples_entrainement_test);
				perceptronTest(&reseau1,exemples_entrainement_test);
				end_of_loop=1;
				break;
			case 'f':
				end_of_loop=1;
				break;

			case '\n':
				break;

			case ' ' :
			 	break;
			default:
			 	printf("\nChoix non disponible :-(\n");
		} 
	} while(!end_of_loop);

	deleteReseau(&reseau1);

	return 0;
}