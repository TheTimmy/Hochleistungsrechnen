/****************************************************************************/
/****************************************************************************/
/**                                                                        **/
/**                TU Muenchen - Institut fuer Informatik                  **/
/**                                                                        **/
/** Copyright: Prof. Dr. Thomas Ludwig                                     **/
/**            Andreas C. Schmidt                                          **/
/**                                                                        **/
/** File:      partdiff-seq.c                                              **/
/**                                                                        **/
/** Purpose:   Partial differential equation solver for Gauss-Seidel and   **/
/**            Jacobi method.                                              **/
/**                                                                        **/
/****************************************************************************/
/****************************************************************************/

/* ************************************************************************ */
/* Include standard header file.                                            */
/* ************************************************************************ */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>
#include <pthread.h>

#include "partdiff-posix.h"

struct calculation_arguments
{
	uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
	uint64_t  num_matrices;   /* number of matrices                             */
	double    h;              /* length of a space between two lines            */
	double    ***Matrix;      /* index matrix used for addressing M             */
	double    *M;             /* two matrices with real values                  */
};

struct calculation_results
{
	uint64_t  m;
	uint64_t  stat_iteration; /* number of current iteration                    */
	double    stat_precision; /* actual precision of all slaves in iteration    */
};

/* ************************************************************************ */
/* Global variables                                                         */
/* ************************************************************************ */

/* time measurement variables */
struct timeval start_time;       /* time when program started                      */
struct timeval comp_time;        /* time when calculation completed                */


/* ************************************************************************ */
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options)
{
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;

	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;
}

/* ************************************************************************ */
/* freeMatrices: frees memory for matrices                                  */
/* ************************************************************************ */
static
void
freeMatrices (struct calculation_arguments* arguments)
{
	uint64_t i;

	for (i = 0; i < arguments->num_matrices; i++)
	{
		free(arguments->Matrix[i]);
	}

	free(arguments->Matrix);
	free(arguments->M);
}

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static
void*
allocateMemory (size_t size)
{
	void *p;

	if ((p = malloc(size)) == NULL)
	{
		printf("Speicherprobleme! (%" PRIu64 " Bytes)\n", size);
		/* exit program */
		exit(1);
	}

	return p;
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static
void
allocateMatrices (struct calculation_arguments* arguments)
{
	uint64_t i, j;

	uint64_t const N = arguments->N;

	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (N + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1));
		}
	}
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static
void
initMatrices (struct calculation_arguments* arguments, struct options const* options)
{
	uint64_t g, i, j;                                /*  local variables for loops   */

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;

	/* initialize matrix/matrices with zeros */
	for (g = 0; g < arguments->num_matrices; g++)
	{
		for (i = 0; i <= N; i++)
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}
	}

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{
			for (i = 0; i <= N; i++)
			{
				Matrix[g][i][0] = 1.0 - (h * i);
				Matrix[g][i][N] = h * i;
				Matrix[g][0][i] = 1.0 - (h * i);
				Matrix[g][N][i] = h * i;
			}

			Matrix[g][N][0] = 0.0;
			Matrix[g][0][N] = 0.0;
		}
	}
}

/*
Struktur, die einem Thread bei seinem start uebergeben werden,
diese enthaelt alle wichtigen informationen ueber die Matrix
und einige vorberechnete variablen.
Des Weiteren wird das maxresiduum hier als pointer uebergeben,
damit dies in den Hauptthread zurueck uebergeben werden kann.
*/
struct ThreadArguments
{
	struct options* options; //die benutzer optionsn
	double pih; //PI * h
	double fpisin; //PI^2/4 * h * h
	double* maxresiduum; //das maximale residuum des threads
	double** Matrix_In; //die eingehende Matrix
	double** Matrix_Out; //die ausgehende Matrix
	int start; //start row
	int end; //end row
	int term_iteration; //aktuelle iteration
};

/*
Die Ausgelagerte calculate funktion, welche werte der Matrix pro thread berechnet.
der void pointer ist ein ThreadArgument mit den enstprechenden eintraegen der Matrix
*/
static
void
runCalculation(void* args)
{
	const struct ThreadArguments* arguments = args;
	const double pih = arguments->pih;
	const double fpisin = arguments->fpisin;
	double fpisin_i = 0.0;
	double star = 0.0;
	double residuum = 0.0;
	struct options const* options = arguments->options;
	const int end = arguments->end;
	const int start = arguments->start;
	const int term_iteration = arguments->term_iteration;
	double** Matrix_In = arguments->Matrix_In;
	double** Matrix_Out = arguments->Matrix_Out;
	double* maxresiduum = arguments->maxresiduum;
	int i, j;

	//berechne vom vorgegeben start zu gegebenem end
	for (i = start; i < end; i++)
	{
		if (options->inf_func == FUNC_FPISIN)
		{
			fpisin_i = fpisin * sin(pih * (double)i);
		}

		/* over all columns */
		for (j = 1; j < i; j++) //halbiere die Matrix, da die anderen Werte irrelevant sind
		{
			star = 0.25 * (Matrix_In[i-1][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + Matrix_In[i+1][j]);

			if (options->inf_func == FUNC_FPISIN)
			{
				star += fpisin_i * sin(pih * (double)j);
			}
			if (options->termination == TERM_PREC || term_iteration == 1)
			{
				residuum = Matrix_In[i][j] - star;
				residuum = (residuum < 0) ? -residuum : residuum;
				*maxresiduum = (residuum < *maxresiduum) ? *maxresiduum : residuum;
			}

			Matrix_Out[i][j] = star;
		}

		//i == j
		star = 0.5 * (Matrix_In[i][i-1] + Matrix_In[i+1][i]);
		if (options->inf_func == FUNC_FPISIN)
		{
			star += fpisin_i * sin(pih * (double)i);
		}
		if(options->termination == TERM_PREC || term_iteration == 1)
		{
			residuum = Matrix_In[i][i] - star;
			residuum = (residuum < 0) ? -residuum : residuum;
			//pruefe den wert des maximums und uebergebe an den pointer wenn notwendig
			*maxresiduum = (residuum < *maxresiduum) ? *maxresiduum : residuum;
		}
		Matrix_Out[i][i] = star;
	}
}

/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options)
{
	int i, j;                                   /* local variables for loops  */
	int m1, m2;                                 /* used as indices for old and new matrices       */
	//double star;                                /* four times center value minus 4 neigh.b values */
	//double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */

	int const N = arguments->N;
	double const h = arguments->h;

	double pih = 0.0;
	double fpisin = 0.0;

	int term_iteration = options->term_iteration;

	/* initialize m1 and m2 depending on algorithm */
	if (options->method == METH_JACOBI)
	{
		m1 = 0;
		m2 = 1;
	}
	else
	{
		m1 = 0;
		m2 = 0;
	}

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	unsigned int threadCount           = (unsigned int) ((int) options->number < N - 1) ? (int) options->number : N - 1;
	double* maximums                   = malloc(sizeof(double) * threadCount);
	pthread_t* threads                 = (pthread_t*) malloc(sizeof(pthread_t) * threadCount);
	struct ThreadArguments* threadArgs = (struct ThreadArguments*) malloc(sizeof(struct ThreadArguments) * threadCount);

	//debug ausgabe
	//printf("Run with %i Threads\n", threadCount);
	int lastStart = 1;
	int k = ((N - 1) * N / 2) / threadCount;
	for(i = 0; i < (int) threadCount; i++)
	{
	        //berechne einige argumente vor
		threadArgs[i].options = (struct options*) options;
		threadArgs[i].pih = pih;
		threadArgs[i].fpisin = fpisin;
		threadArgs[i].maxresiduum = &maximums[i];
		threadArgs[i].Matrix_In = NULL;
		threadArgs[i].Matrix_Out = NULL;

		//gibt die start row des aktuellen arguments an
		threadArgs[i].start = lastStart;
		//berechne dessen ende
		threadArgs[i].end   = 0.5 * (sqrt(8 * k * (i+1) + 1) + 1); // besserer lasten ausgleich als ((float)N) / ((float) threadCount) * (i + 1) + 1;
		//pruefe ob das ende groesser ist als die maximale Anzahl der rows
		threadArgs[i].end   = (threadArgs[i].end > N) ? N : threadArgs[i].end;
		//die neue start row ist das ende dieses threads
		lastStart = threadArgs[i].end;

		//debug ausgabe
		//printf("From start %i to end %i with N: %i\n", threadArgs[i].start, threadArgs[i].end, N);
		threadArgs[i].term_iteration = 0;
	}

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];

 		maxresiduum = 0;
		/* over all rows */
		for(i = 0; i < (int) threadCount; i++)
		{
			maximums[i] = 0;
			threadArgs[i].term_iteration = term_iteration;
			threadArgs[i].Matrix_In = Matrix_In;
			threadArgs[i].Matrix_Out = Matrix_Out;
			//starte einen neuen thread, dies ist hier eigentlich unguenstig, aber
			//da die implementierung eines threadpools zu aufwendig ist, wird hier
			//darauf verzichtet, und die naive methode benutzt
			pthread_create(&threads[i], NULL, (void* (*)(void*)) &runCalculation, (void*)(&threadArgs[i]));
		}

		for(i = 0; i < (int) threadCount; i++)
		{
			//fuehrt den thread mit dem Hauptthread zusammen, dies geschieht erst,
			//wenn der thread zu ende gerechnet hat
			pthread_join(threads[i], NULL);
			//pruefen ob das maximums des thrseadsgroesser ist als das aktuelle
			maxresiduum = (maximums[i] < maxresiduum) ? maxresiduum : maximums[i];
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation, depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if (maxresiduum < options->term_precision)
			{
				term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}
	}

	free(threadArgs);
	free(maximums);
	free(threads);

	double** Matrix = arguments->Matrix[m2];
	for(i = 1; i < N; i++)
	{
		for(j = 1; j < i; j++)
		{
			Matrix[j][i] = Matrix[i][j];
		}
	}
	results->m = m2;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static
void
displayStatistics (struct calculation_arguments const* arguments, struct calculation_results const* results, struct options const* options)
{
	int N = arguments->N;
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;

	printf("Berechnungszeit:    %f s \n", time);
	printf("Speicherbedarf:     %f MiB\n", (N + 1) * (N + 1) * sizeof(double) * arguments->num_matrices / 1024.0 / 1024.0);
	printf("Berechnungsmethode: ");

	if (options->method == METH_GAUSS_SEIDEL)
	{
		printf("Gauss-Seidel");
	}
	else if (options->method == METH_JACOBI)
	{
		printf("Jacobi");
	}

	printf("\n");
	printf("Interlines:         %" PRIu64 "\n",options->interlines);
	printf("Stoerfunktion:      ");

	if (options->inf_func == FUNC_F0)
	{
		printf("f(x,y) = 0");
	}
	else if (options->inf_func == FUNC_FPISIN)
	{
		printf("f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)");
	}

	printf("\n");
	printf("Terminierung:       ");

	if (options->termination == TERM_PREC)
	{
		printf("Hinreichende Genaugkeit");
	}
	else if (options->termination == TERM_ITER)
	{
		printf("Anzahl der Iterationen");
	}

	printf("\n");
	printf("Anzahl Iterationen: %" PRIu64 "\n", results->stat_iteration);
	printf("Norm des Fehlers:   %e\n", results->stat_precision);
	printf("\n");
}

/****************************************************************************/
/** Beschreibung der Funktion DisplayMatrix:                               **/
/**                                                                        **/
/** Die Funktion DisplayMatrix gibt eine Matrix                            **/
/** in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.   **/
/**                                                                        **/
/** Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix    **/
/** ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie   **/
/** sieben Zwischenzeilen ausgegeben.                                      **/
/****************************************************************************/
static
void
DisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	int x, y;

	double** Matrix = arguments->Matrix[results->m];

	int const interlines = options->interlines;

	printf("Matrix:\n");

	for (y = 0; y < 9; y++)
	{
		for (x = 0; x < 9; x++)
		{
			printf ("%7.4f", Matrix[y * (interlines + 1)][x * (interlines + 1)]);
		}

		printf ("\n");
	}

	fflush (stdout);
}

/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int
main (int argc, char** argv)
{
	struct options options;
	struct calculation_arguments arguments;
	struct calculation_results results;

	/* get parameters */
	AskParams(&options, argc, argv);              /* ************************* */

	initVariables(&arguments, &results, &options);           /* ******************************************* */

	allocateMatrices(&arguments);        /*  get and initialize variables and matrices  */
	initMatrices(&arguments, &options);            /* ******************************************* */

	gettimeofday(&start_time, NULL);                   /*  start timer         */
	calculate(&arguments, &results, &options);                                      /*  solve the equation  */
	gettimeofday(&comp_time, NULL);                   /*  stop timer          */

	displayStatistics(&arguments, &results, &options);
	DisplayMatrix(&arguments, &results, &options);

	freeMatrices(&arguments);                                       /*  free memory     */

	return 0;
}
