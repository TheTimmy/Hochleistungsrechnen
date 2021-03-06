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


//Openmp includieren, damit alle Funktionen gefunden werden
#if(OPENMP)
#include <omp.h>
#endif

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff-seq.h"

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
	uint64_t g, i; //j ist nicht wichtig                                /*  local variables for loops   */

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;

	//Das initialize wird durch das memset ersetzt, was ein wenig schneller schneller funktioniert
	/* initialize matrix/matrices with zeros */
	/*for (g = 0; g < arguments->num_matrices; g++)
	{
		for (i = 0; i <= N; i++)
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}
	}*/
#ifdef OPENMP
	//wenn Openmp untuetzt wird, dann soll die besetzung der Randwerte auch mit mehreren
	//Treads gemacht werden.
	omp_set_num_threads(options->number);
#endif
	memset(**Matrix, 0, sizeof(double) * (N + 1) * (N + 1));

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{
#ifdef OPENMP
			//alle werte mittels eines extra threads setzen, dabei ist i private,
			//da alle threads ihre eigene Laufvariable haben, g und h werden
			//auch als private gesetzt, behalten aber ihren vorherigen wert.
			#pragma omp parallel for private(i) firstprivate(g, h) shared(Matrix)
#endif
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

	//if weggelassen, da diese berechnungen nur einmalig laufen,
	//und ein sprung mittels brach-prediction genauso lange dauern wuerde.
	const double piH = PI * h;
	const double piSqrH = 0.25 * TWO_PI_SQUARE * h * h;

#ifdef OPENMP
	//setzen der threads expilziet (der uebersichts halber, alternativ ginge es auch mit num_threads in der pragma clause)
	omp_set_num_threads(options->number);
#endif

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];
		maxresiduum = 0;
#ifdef OPENMP
	#ifdef OVER_BASE
		/*
		Parallelisiergg ueber die Spalten.
		hierbei wird jedem Thread seine eigene i,j Variable zugewiesen.
		Zusaetzlich bekommt jeder thread die bereits berechneten werte piH und piSqrH zugewiesen,
		damit diese nicht wieder innerhalb der schleife berechnet werden muessen.
		Des Weiteren werden die Variablen Matrix_In, Matrix_Out, term_iteration, options noch ueber die
		Threads verteilt, letztendlich werden alle anderen Variablen mit der default(none) klausel
		ausgeschlossen.
		Die Klausel reduction(max:maxresiduum) verhindert, das eine critical Klausel innerhalb der
		Schleife notwendig wird, somit wird dort etwas rechenzeit gespart.
		*/

		/* einfache paralleisierung ueber die rows */
		#pragma omp parallel for private(i, j) firstprivate(piH, piSqrH) shared(Matrix_In, Matrix_Out, term_iteration, options) reduction(max:maxresiduum) schedule(SCHEDULE_TYPE, SCHEDULE_SIZE) default(none)
		for (i = 1; i < N; i++) {
	#endif
	#ifdef OVER_COL
		//Das gleiche gilt hier wie fuer OVER_BASE
		/* Only parallize over the rows*/
		#pragma omp parallel for private(i, j) firstprivate( piH, piSqrH) shared(Matrix_Out, Matrix_In, term_iteration, options) reduction(max:maxresiduum) schedule(SCHEDULE_TYPE, SCHEDULE_SIZE) default(none)
		for (i = 1; i < N; i++) {
	#endif
	#ifdef OVER_ELEM
		/*
		Bei OVER_ELEM musste die Schleife abgeaendert werden, damit alle elemente expliziet durch gelaufen werden.
		Sonst wuerden nur die Zeilen abgebeitet werden. Damit ergiebt sich dieses Konstrukt.
		Allerdings kann hier keine Optimierung wie bei OVER_COL oder OVER_BASE angewand werden.
		*/

		int k = 0;
		/* Only parallize over the elements */
		#pragma omp parallel for private(i, j, k) firstprivate(piH, piSqrH) shared(Matrix_Out, Matrix_In, term_iteration, options) reduction(max:maxresiduum) schedule( SCHEDULE_TYPE, SCHEDULE_SIZE) default(none)
		for (k = 0; k < N * (N - 1); k++) {
			i = k / N + 1;
			j = k % (N - 1) + 1;
	#endif
	#ifdef OVER_ROW
		/*
		Hier muss die addressierung der Daten veraendert werden, da sonst wieder die Zeilen, und nicht
		die Spalten abgearbeitet werden. Dies fuehrt aber zu cache problemen, da dieser nicht optimal genutzt werden
		kann.
		*/
		#pragma omp parallel for private(j, i) firstprivate(piH, piSqrH) shared(Matrix_Out, Matrix_In, term_iteration, options) reduction(max:maxresiduum) schedule(SCHEDULE_TYPE, SCHEDULE_SIZE) default(none) //static, 8)
		for (j = 1; j < N; j++) {
	#endif
#else
		//Hier wird sonst die Basis implementierung abgelaufen
		for (i = 1; i < N; i++) {
#endif
			/*
			Da die Matrix Matrix_In nur read only ist, kann diesueber mehrere Threads
			verteilt werden.
			*/
#ifndef OVER_ROW
			const double* rowSub1 = Matrix_In[i-1];
			const double* row     = Matrix_In[i];
			const double* rowAdd1 = Matrix_In[i+1];
#endif

			//Da fpisin nur im scope des Threas besteht, kann hier einfach in fpisin geschrieben werden.
			double fpisin = 0;
			if (options->inf_func == FUNC_FPISIN)
			{
#ifndef OVER_ROW
				//fuer die Row muss die Implementierung entsprechend geaendert werden.
				fpisin = piSqrH * sin(piH * (double)i);
#else
				fpisin = piSqrH * sin(piH * (double)j);
#endif
			}

			/* over all columns */
#ifdef OPENMP
	#ifdef OVER_ROW
			//Hier werden nun die Zeilen durchlaufen
			for (i = 1; i < N; i++)
	#elif !defined(OVER_ELEM)
			//Da die for Schleife der OVER_ELEM implementierung bereits alles ablaueft duerfen hier keine
			//weiteren Schleifen durchlaufen werden.
			for (j = 1; j < i; j++)
	#endif
#else
			for (j = 1; j < N; j++)
#endif
			{
#ifdef OVER_ROW
        	                const double* rowSub1 = Matrix_In[i-1];
	                        const double* row     = Matrix_In[i];
                	        const double* rowAdd1 = Matrix_In[i+1];
#endif
				/* Entferne auch star aus dem Hauptscope, da es sonst zu lost updates kommen kann*/
				double star = 0.25 * (rowSub1[j] + row[j-1] + row[j+1] + rowAdd1[j]);
				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin * sin(piH * (double)j);
				}

				if (options->termination == TERM_PREC || term_iteration == 1)
				{
					double residuum = row[j] - star;
	                                residuum = fabs(residuum); //(residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}
				Matrix_Out[i][j] = star;
			}
#if defined(OPENMP)
	#ifdef OVER_ROW
                        const double* row     = Matrix_In[j];
                        const double* rowAdd1 = Matrix_In[j+1];
			i = j;
	#endif
			// i == j
			double star = 0.5 * (row[i - 1] + rowAdd1[i]);
			if (options->inf_func == FUNC_FPISIN)
                        {
	                        star += fpisin * sin(piH * (double)i);
                        }
                        if (options->termination == TERM_PREC || term_iteration == 1)
                        {
                        	double residuum = row[i] - star;
                                residuum = fabs(residuum); //(residuum < 0) ? -residuum : residuum;
                                maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}

			Matrix_Out[i][i] = star;
#endif
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
	} //end while

#if defined(OPENMP)
	double*** Matrix = arguments->Matrix;
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < i; j++)
		{
			Matrix[m2][j][i] = Matrix[m2][i][j];
		}
	}
#endif

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
	AskParams(&options, argc, argv);                  /* ******************************************* */

	initVariables(&arguments, &results, &options);    /* ******************************************* */

	allocateMatrices(&arguments);                     /*  get and initialize variables and matrices  */
	initMatrices(&arguments, &options);               /* ******************************************* */

	gettimeofday(&start_time, NULL);                  /*  start timer         */
	calculate(&arguments, &results, &options);        /*  solve the equation  */
	gettimeofday(&comp_time, NULL);                   /*  stop timer          */

	displayStatistics(&arguments, &results, &options);
	DisplayMatrix(&arguments, &results, &options);

	freeMatrices(&arguments);                         /*  free memory     */

	return 0;
}
