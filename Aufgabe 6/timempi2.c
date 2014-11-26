#define _BSD_SOURCE //entfernt die implizite definition von gethostname

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define HOSTNAMELENGTH 64
#define TIMESTAMPLENGTH 64
#define STAMPLENGTH 128

#define checkError(val,string) if(val) {printf(string); exit(-1);}

int main(int argc, char** argv)
{
	//Basis initilisierung von mpi
	checkError(MPI_Init(&argc, &argv) != MPI_SUCCESS, "Cound not init mpi");

	//den rank des prozesses aus mpi lesen
	int rank = 0, processCount = 0;
	int microsec = 0;
	int microseconds = 0;
	checkError(MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS, "Cound not get the rank of the process");

	//die anzahl der prozesse herausfinden
      	checkError(MPI_Comm_size(MPI_COMM_WORLD, &processCount) != MPI_SUCCESS, "Could not get the Comm_size");

	//und ein Array erstellen, was den namen des knoten enthalten soll
	char* namestr = NULL;
	char* timestr = NULL;
	char* string  = NULL;
	char* buffer  = NULL;

	//wenn es nicht der master prozess ist ein fach zwei arrays mit passender groesse erstellen
	namestr = malloc(sizeof(char) * HOSTNAMELENGTH);
	checkError(namestr == NULL, "Could not allocate enouth memory for the name");
	timestr = malloc(sizeof(char) * TIMESTAMPLENGTH);
	checkError(timestr == NULL, "Could not allocate enouth memory for the time");
	string = malloc(sizeof(char) * STAMPLENGTH);
	checkError(string == NULL, "Could not allocate enouth memory for the buffer");

	if(rank == 0)
	{
		//da dies der Master prozess ist die anzahl aller Prozesse herrausfinden und
		//den noetigen Speicher fuer die ergebnisse allokieren
		buffer = malloc(sizeof(char) * STAMPLENGTH * processCount);
		checkError(buffer == NULL, "Could not allocate enouth memory for the buffer");
	}

	//den hostname auslesen
	checkError(gethostname(namestr, HOSTNAMELENGTH) != 0, "Could not get hostname");

	//die systemzeit des hostsystems auslesen
	struct timeval time;
	checkError(gettimeofday(&time, NULL) != 0, "Could not get the current time");
	strftime(timestr, TIMESTAMPLENGTH, "%Y-%m-%d %T", localtime(&time.tv_sec));
	//den eigentlichen string zusammen bauen mit rank als indikatoruer die richtige reihenfolge
	sprintf(string, "%s: %s.%ld\n", namestr, timestr, time.tv_usec);

	microsec = time.tv_usec;
	if(rank == 0)
	{
		microsec = INT_MAX;
	}

	checkError(MPI_Reduce(&microsec, &microseconds, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD) != MPI_SUCCESS, "Fail to reduce the minimum time");

	//auf die Informationen der einzelnen Prozesse warten, hierbei wird ein string der groesse STAMPLENGTH gesendet
	//und auch empfangen. Die daten landen dann jeweils in buffer.
	checkError(MPI_Gather(string, STAMPLENGTH, MPI_CHAR, buffer, STAMPLENGTH, MPI_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS, "Fail to gather informations");
	if(rank == 0)
	{
		//da dies der hauptprozess ist daten ausgeben
		int i = 0;
		for(i = 1; i < processCount; i++)
		{
			printf("%.*s", STAMPLENGTH, buffer + STAMPLENGTH * i);
		}

		printf("%i\n", microseconds);
		fflush(stdout);
	}

	checkError(MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS, "Fail to wait for all processes");
	printf("Rank %i beendet jetzt!\n", rank);
	fflush(stdout);

	//alle allocierten daten frei geben.
	free(namestr);
	free(timestr);
	free(string);
	free(buffer);

	//beenden
	checkError(MPI_Finalize() != MPI_SUCCESS, "Could not finalize the process");
	return 0;
}
