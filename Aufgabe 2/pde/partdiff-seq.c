/****************************************************************************/
/****************************************************************************/
/**                                                                        **/
/**                TU Muenchen - Institut fuer Informatik                  **/
/**                                                                        **/
/** Copyright: Prof. Dr. Thomas Ludwig                                     **/
/**            Andreas C. Schmidt                                          **/
/**            JK und andere besseres Timing, FLOP-Berechnung              **/
/**                                                                        **/
/** File:      partdiff-seq.c                                              **/
/**                                                                        **/
/** Purpose:   Partial differential equation solver for Gauss-Seidel and   **/
/**            Jacobi methods.                                             **/
/**                                                                        **/
/****************************************************************************/
/****************************************************************************/

/* ************************************************************************ */
/* Include standard header file.                                            */
/* ************************************************************************ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h> //added for memset
#include <malloc.h>
#include <sys/time.h>
#include "partdiff-seq.h"

struct calculation_arguments
{
	int     N;               /* number of spaces between lines (lines=N+1)     */
	int     num_matrices;    /* number of matrices                             */
	double  ***Matrix;       /* index matrix used for addressing M             */
	double  *M;              /* two matrices with real values                  */
	double  h;               /* length of a space between two lines            */
};

struct calculation_results
{
	int     m;
	int     stat_iteration; /* number of current iteration                    */
	double  stat_precision; /* actual precision of all slaves in iteration    */
};

/* ************************************************************************ */
/* Forward declerations							    */
/* ************************************************************************ */
double getResiduum (struct calculation_arguments* arguments, struct options* options, int x, int y);


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
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	arguments->N = options->interlines * 8 + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = (float)( ( (float)(1) ) / (arguments->N));

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
	int i;
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
		printf("\n\nSpeicherprobleme!\n");
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
	int i, j;

	int N = arguments->N;
	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (N + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));
	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = (double*)(arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1)));
		}
	}
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static
void
initMatrices (struct calculation_arguments* arguments, struct options* options)
{
	int i, j;                                /*  local variables for loops   */

	int N = arguments->N;
	double h = arguments->h;
	double*** Matrix = arguments->Matrix;

	/* initialize matrix/matrices with zeros */
	memset(**Matrix, 0, sizeof(double) * (N + 1) * (N + 1) * arguments->num_matrices);

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for(j = 0; j < arguments->num_matrices; j++)
		{
			for (i = 0; i <= N; i++)
			{
				Matrix[j][i][0] = 1 - (h * i);
				Matrix[j][i][N] = h * i;
				Matrix[j][0][i] = 1 - (h * i);
				Matrix[j][N][i] = h * i;
			}

			Matrix[j][N][0] = 0;
			Matrix[j][0][N] = 0;
		}
	}
}

/* ************************************************************************ */
/* getResiduum: calculates residuum                                         */
/* Input: x,y - actual column and row                                       */
/* ************************************************************************ */
double
getResiduum (struct calculation_arguments* arguments, struct options* options, int x, int y)
{
	switch(options->inf_func)
	{
	case FUNC_F0:
		return 0;
	break;
	default:
		return TWO_PI_SQUARE * sin((double)(y) * PI * arguments->h) * sin((double)(x) * PI * arguments->h) * arguments->h * arguments->h;
	break;
	}
}

/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments* arguments, struct calculation_results *results, struct options* options)
{
	int i, j;                                   /* local variables for loops  */
	int m1, m2;                                 /* used as indices for old and new matrices       */
	double star;                                /* four times center value minus 4 neigh.b values */
	double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */

	int N = arguments->N;
	double*** Matrix = arguments->Matrix;

	/* initialize m1 and m2 depending on algorithm */
	if (options->method == METH_GAUSS_SEIDEL)
	{
		m1=0; m2=0;
	}
	else
	{
		m1=0; m2=1;
	}

	//precompute the residuum
	double* residuumMatrix = (double*) allocateMemory(sizeof(double) * N * (N + 1));
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < i + 1; j++)
		{
			residuumMatrix[j + i * N] = getResiduum(arguments, options, i, j);
		}
	}

	while (options->term_iteration > 0)
	{
		maxresiduum = 0;

		/* over all rows */
		double** mat1 = Matrix[m1];
		double** mat2 = Matrix[m2];
		for (i = 1; i < N; i++)
		{
			double* row1 = mat1[i];
			double* row2 = mat2[i];
			double* row2Sub1 = mat2[i - 1];
			double* row2Add1 = mat2[i + 1];
			/* over the half of all columns (to i) because A = A^t */
			for (j = 1; j < i; j++)
			{
				star = -row2Sub1[j] - row2[j-1] - row2[j + 1] - row2Add1[j] + 4 * row2[j];
				residuum = (residuumMatrix[j + i * N] - star) / 4.0; //getResiduum(arguments, options, i, j, star);
				row1[j] = row2[j] + residuum;
				residuum = fabs((double)residuum); //(residuum < 0) ? -residuum : residuum;
				maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}

			//i == j
			star = -2 * (row2[i - 1] + row2Add1[i]) + 4 * row2[i];
			residuum = (residuumMatrix[i + i * N] - star) / 4.0;
			row1[i] = row2[i] + residuum;
			residuum = fabs((double)residuum); //(residuum < 0) ? -residuum : residuum;
			maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i=m1; m1=m2; m2=i;

		/* check for stopping calculation, depending on termination method */
		switch(options->termination)
		{
			case TERM_PREC:
				if(maxresiduum < options->term_precision)
					options->term_iteration = 0;
			break;
			case TERM_ITER:
				options->term_iteration--;
			break;
		}
	}

	//because the matrix has the property that A^T = A the computed half must be copied
	double** mat = Matrix[m2];
	for(j = 0; j < N; j++)
	{
		for(i = 0; i < N; i++)
			mat[j][i] = mat[i][j];
	}


	results->m = m2;
	free(residuumMatrix);
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static
void
displayStatistics (struct calculation_arguments* arguments, struct calculation_results *results, struct options* options)
{
	int N = arguments->N;

	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;
	printf("Berechnungszeit:    %f s \n", time);

	//Calculate Flops
	// star op = 5 ASM ops (+1 XOR) with -O3, matrix korrektur = 1
	double q = 6;
	double mflops;

	if (options->inf_func == FUNC_F0)
	{
		// residuum: checked 1 flop in ASM, verified on Nehalem architecture.
		q += 1.0;
	}
	else
	{
		// residuum: 11 with O0, but 10 with "gcc -O3", without counting sin & cos
		q += 10.0;
	}

	/* calculate flops  */
	mflops = (q * (N - 1) * (N - 1) * results->stat_iteration) * 1e-6;
	printf("Executed float ops: %f MFlop\n", mflops);
	printf("Speed:              %f MFlop/s\n", mflops / time);

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
	printf("Interlines:         %d\n",options->interlines);
	printf("Stoerfunktion:      ");

	if (options->inf_func == FUNC_F0)
	{
		printf("f(x,y)=0");
	}
	else if (options->inf_func == FUNC_FPISIN)
	{
		printf("f(x,y)=2pi^2*sin(pi*x)sin(pi*y)");
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
	printf("Anzahl Iterationen: %d\n", results->stat_iteration);
	printf("Norm des Fehlers:   %e\n", results->stat_precision);
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

	displayStatistics(&arguments, &results, &options);                                  /* **************** */
	DisplayMatrix("Matrix:",                              /*  display some    */
			arguments.Matrix[results.m][0], options.interlines);            /*  statistics and  */

	freeMatrices(&arguments);                                       /*  free memory     */

	return 0;
}
