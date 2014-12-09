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

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff-par.h"

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
	arguments->globalN = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = 2;
	arguments->h = 1.0 / arguments->globalN;

	if(arguments->commSize == 1)
	{
	  printf("Only one Process\n");
	  arguments->start = 0;
	  arguments->end   = arguments->globalN;
	  arguments->N     = arguments->globalN;
	}
        else
	{
	  uint64_t buffer[3];
	  if(arguments->rank == 0)
	  {
	    int range = arguments->globalN / arguments->commSize;
	    int start = 0;
	    int end = range;
	    arguments->start = start;
	    arguments->end   = end;
	    arguments->N     = end - start;
	  
	    for(int i = 1; i < arguments->commSize - 1; i++)
	    {
	      start = end;
	      end += range;
	      
	      buffer[0] = start - 1;
	      buffer[1] = end;
	      MPI_Send(buffer, 2, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
	    }

	    start = end - 1;
	    end = arguments->globalN;
	    buffer[0] = start;
	    buffer[1] = end;
	    MPI_Send(buffer, 2, MPI_UINT64_T, arguments->commSize - 1, 0, MPI_COMM_WORLD);
	  }
	  else
	  {
	    MPI_Recv(buffer, 2, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    arguments->start = buffer[0];
	    arguments->end   = buffer[1];
	    arguments->N     = arguments->end - arguments->start;
	  }
	}	

	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;

	printf("Rank: %i, go from: %"PRIu64 " to %" PRIu64 " with N: %" PRIu64 " and Global N %"PRIu64 "\n",
	       arguments->rank, arguments->start, arguments->end, arguments->N, arguments->globalN);
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
	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (arguments->globalN + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (arguments->globalN + 1) * (N + 1)) + (j * (arguments->globalN + 1));
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
			for (j = 0; j <= arguments->globalN; j++)
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
	    for (i = 0; i < N + 1; i++)
	    {
	      Matrix[g][i][0] = 1.0 - (h * (i + arguments->start));
	      Matrix[g][i][arguments->globalN] = h * (i + arguments->start);
	    }
	  }
 
	  if(arguments->rank == 0)
	  {
	    for (g = 0; g < arguments->num_matrices; g++)
	    {
	      for(int i = 0; i < (int) arguments->globalN; i++)
	      {
		Matrix[g][0][i] = 1 - h * i;
	      }
	    }
	  }

	  if(arguments->rank == arguments->commSize - 1)
	  {
	    for(g = 0; g < arguments->num_matrices; g++)
	    {
	      for(int i = 0; i < (int) arguments->globalN; i++)
	      {
		Matrix[g][arguments->N][i] = h * i;
	      }
	    }
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
	double star;                                /* four times center value minus 4 neigh.b values */
	double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */
  
	int const N = arguments->N;
	double const h = arguments->h;

	double pih = 0.0;
	double fpisin = 0.0;

	int term_iteration = options->term_iteration;
	m1 = 0;
	m2 = 1;

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];

		maxresiduum = 0;

		/* over all rows */
		for (i = 1; i < N; i++)
		{
			double fpisin_i = 0.0;

			double posi = i + ((double)arguments->start);
			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)posi);
			}

#ifdef HALFMATRIX
			for (j = 1; j < i; j++)
#else
			  for (j = 1; j < (int) arguments->globalN; j++)
#endif
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
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}

				Matrix_Out[i][j] = star;
			}

#ifdef HALFMATRIX
			star = 0.25 * (Matrix_In[i][i-1] + Matrix_In[i+1][i]);
			if (options->inf_func == FUNC_FPISIN)
			{
			  star += fpisin_i * sin(pih * (double)posi);
			}
			if (options->termination == TERM_PREC || term_iteration == 1)
			{
			  residuum = Matrix_In[i][i] - star;
			  residuum = (residuum < 0) ? -residuum : residuum;
			  maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}
     
			Matrix_Out[i][i] = star;
#endif
		}

		if(arguments->rank > 0)
		{
		  MPI_Sendrecv(Matrix_Out[1], arguments->globalN + 1, MPI_DOUBLE, arguments->rank - 1, arguments->rank,
			       Matrix_Out[0], arguments->globalN + 1, MPI_DOUBLE, arguments->rank - 1, arguments->rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if(arguments->rank != arguments->commSize - 1)
		{
		  MPI_Sendrecv(Matrix_Out[N - 1], arguments->globalN + 1, MPI_DOUBLE, arguments->rank + 1, arguments->rank,
			       Matrix_Out[N], arguments->globalN + 1, MPI_DOUBLE, arguments->rank + 1, arguments->rank + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		//check for residuum
		if(options->termination == TERM_PREC || term_iteration == 1)
		{ 
		  double currentResiduum = maxresiduum; 
		  MPI_Allreduce(&maxresiduum, &currentResiduum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		  maxresiduum = currentResiduum;
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

static
void
DebugDisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	int x, y;

	double** Matrix = arguments->Matrix[results->m];
	int const interlines = options->interlines;
	printf("Matrix:\n");

	for (y = 0; y < (int) arguments->N + 1; y++)
	{
	  for (x = 0; x < (int) arguments->globalN + 1; x++)
		{
			printf ("%7.4f", Matrix[y][x]);
		}

		printf ("\n");
	}

	fflush (stdout);
}

static
void
DisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options, int rank, int size, int from, int to)
{
  int const elements = 8 * options->interlines + 9;

  int x, y;
  double** Matrix = arguments->Matrix[results->m];
  MPI_Status status;

  /* first line belongs to rank 0 */
  if (rank == 0)
    from--;

  /* last line belongs to rank size - 1 */
  if (rank + 1 == size)
    to++;

  if (rank == 0)
    printf("Matrix:\n");

  //printf("rank %i from %i to %i\n", rank, from, to);
  for (y = 0; y < 9; y++)
  {
    int line = y * (options->interlines + 1);

    if (rank == 0)
    {
      /* check whether this line belongs to rank 0 */
      if (line < from || line > to)
      {
	//if(rank == 0)
	//  printf("Need line: %i, y=%i\n", line, y);
        /* use the tag to receive the lines in the correct order
         * the line is stored in Matrix[0], because we do not need it anymore */
        MPI_Recv(Matrix[0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      if (line >= from && line <= to)
      {
	//printf("Send line %i, rank %i, y=%i\n", line, arguments->rank, y);
        /* if the line belongs to this process, send it to rank 0
         * (line - from + 1) is used to calculate the correct local address */
        MPI_Send(Matrix[line - from + 1], elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
      }
    }

    if (rank == 0)
    {
      //if(rank == 0)
      //  printf("Print line: %i, y=%i\n", line, y);

      for (x = 0; x < 9; x++)
      {
        int col = x * (options->interlines + 1);
	
        if (line >= from && line <= to)
        {
          /* this line belongs to rank 0 */
          printf("%7.4f", Matrix[line][col]);
        }
        else
        {
          /* this line belongs to another rank and was received above */
          printf("%7.4f", Matrix[0][col]);
        }
      }

      printf("\n");
    }
  }

  fflush(stdout);
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

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &arguments.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &arguments.commSize);

	/* get parameters */
	AskParams(&options, argc, argv, arguments.rank);              /* ************************* */
	if(options.interlines * 8 + 9 - 1 < arguments.commSize)
	{
	  if(arguments.rank == 0)
	    printf("\n\n\n\tToo many processes for this small matrix, waste of resources and bugs can appear!\n\tPlease restart with a fewer count of processes.\n\tExit now.\n\n\n\n");
	  MPI_Abort(MPI_COMM_WORLD, - 1);
	}

	initVariables(&arguments, &results, &options);           /* ******************************************* */

	allocateMatrices(&arguments);
	initMatrices(&arguments, &options);

	gettimeofday(&start_time, NULL);              
	calculate(&arguments, &results, &options);      
	gettimeofday(&comp_time, NULL);

	if(arguments.rank == 0) 
	{
	  displayStatistics(&arguments, &results, &options);
	}

	int add = (arguments.rank > 0) ? 1 : 0;
	int sub = (arguments.rank < arguments.commSize) ? 1 : 0;

	DisplayMatrix(&arguments, &results, &options, arguments.rank, arguments.commSize, arguments.start + add, arguments.end - sub);
	freeMatrices(&arguments);

	//printf("Process: %i exit now\n", arguments.rank);
	MPI_Finalize();
	return 0;
}
