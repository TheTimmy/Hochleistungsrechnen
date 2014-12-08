#include "mpi.h"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff-par.h"

#define checkError(x, str) { if(!x) { printf(str); MPI_Abort(MPI_COMM_WORLD, -1); } }

struct calculation_arguments
{
  int start, end;
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

typedef struct MPI_Options
{
  int rank;
  int size;
} MPI_options;

/* time measurement variables */
struct timeval start_time;       /* time when program started                      */
struct timeval comp_time;        /* time when calculation completed                */

static
void 
initMPI(int argc, char** argv, MPI_options* options)
{
  checkError(MPI_Init(&argc, &argv) == MPI_SUCCESS, "Could not init mpi");
  checkError(MPI_Comm_rank(MPI_COMM_WORLD, &(options->rank)) == MPI_SUCCESS, "Could not get the rank");
  checkError(MPI_Comm_size(MPI_COMM_WORLD, &(options->size)) == MPI_SUCCESS, "Could not the size");
}

static
const int
getMatrixSize(struct options const*  options)
{
  return (options->interlines * 8) + 9 - 1;
}

static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options, const MPI_options* mpiOptions)
{
  const int N = getMatrixSize(options);
  const int rank = mpiOptions->rank;
  const int size = mpiOptions->size;
  arguments->num_matrices = 2;
  results->m = 0;
  results->stat_iteration = 0;
  results->stat_precision = 0;
  arguments->h = 1.0 / N;

  int* buffer = (int*) malloc(sizeof(int) * 2);
  if(rank == 0)
  {
    const int range = N / size;
    int start = 0;
    int end = range;
    arguments->N = end - start;
    arguments->start = start;
    arguments->end = end;

    for(int i = 1; i < size - 1; i++)
    {
      start = end;
      end += range;
      
      buffer[0] = start;
      buffer[1] = end;
      MPI_Send(buffer, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    start = end;
    end = N;
    buffer[0] = start;
    buffer[1] = end;
    MPI_Send(buffer, 2, MPI_INT, size - 1, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Recv(buffer, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    arguments->N = buffer[1] - buffer[0];
    arguments->start = buffer[0];
    arguments->end = buffer[1];
  }
 
  printf("Hallo from %i, with start: %i and end %i\n", rank, arguments->start, arguments->end);
 
  free(buffer);
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * rank and size are the MPI rank and size, respectively.
 * from and to denote the global(!) range of lines that this process is responsible for.
 *
 * Example with 9 matrix lines and 4 processes:
 * - rank 0 is responsible for 1-2, rank 1 for 3-4, rank 2 for 5-6 and rank 3 for 7.
 *   Lines 0 and 8 are not included because they are not calculated.
 * - Each process stores two halo lines in its matrix (except for ranks 0 and 3 that only store one).
 * - For instance: Rank 2 has four lines 0-3 but only calculates 1-2 because 0 and 3 are halo lines for other processes. It is responsible for (global) lines 5-6.
 */
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

  for (y = 0; y < 9; y++)
  {
    int line = y * (options->interlines + 1);

    if (rank == 0)
    {
      /* check whether this line belongs to rank 0 */
      if (line < from || line > to)
      {
        /* use the tag to receive the lines in the correct order
         * the line is stored in Matrix[0], because we do not need it anymore */
        MPI_Recv(Matrix[0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      if (line >= from && line <= to)
      {
        /* if the line belongs to this process, send it to rank 0
         * (line - from + 1) is used to calculate the correct local address */
        MPI_Send(Matrix[line - from + 1], elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
      }
    }

    if (rank == 0)
    {
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

	const int size = arguments->end - arguments->start;
	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (size + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
		  arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (size + 1)) + (j * (N + 1));
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
	uint64_t g, i;
	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;

	int size = arguments->end - arguments->start;
	memset(**Matrix, 0, sizeof(double) * (N + 1) * size);

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{
			for (i = 0; i <= size; i++)
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

	const int size = arguments->start - arguments->end;
	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];

		maxresiduum = 0;

		/* over all rows */
		for (i = 1; i < size; i++)
		{
			double fpisin_i = 0.0;

			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)i);
			}

			/* over all columns */
			for (j = 1; j < i; j++) //
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

			//i == j
			star = 0.5 * (Matrix_In[i][i-1] + Matrix_In[i+1][i]);
			if(options->inf_func == FUNC_FPISIN)
			{
				star += fpisin_i * sin(pih * (double)i);
			}
			if(options->termination == TERM_PREC || term_iteration == 1)
			{
				residuum = Matrix_In[i][i] - star;
				residuum = (residuum < 0) ? -residuum : residuum;
				maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}
			Matrix_Out[i][i] = star;
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

	double** Matrix = arguments->Matrix[m2];
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < i; j++)
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

int main(int argc, char** argv)
{
	struct options options;
	struct calculation_arguments arguments;
	struct calculation_results results;

	AskParams(&options, argc, argv); 

	MPI_options mpiOptions;
	initMPI(argc, argv, &mpiOptions);
	initVariables(&arguments, &results, &options, &mpiOptions);

	allocateMatrices(&arguments); 
	initMatrices(&arguments, &options);

	gettimeofday(&start_time, NULL); 
	calculate(&arguments, &results, &options);
	gettimeofday(&comp_time, NULL); 

	if(mpiOptions.rank == 0)
	{
	    displayStatistics(&arguments, &results, &options);
	}

	DisplayMatrix(&arguments, &results, &options, mpiOptions.rank, mpiOptions.size, arguments.start, arguments.end);
	freeMatrices(&arguments);
	MPI_Finalize();

	return 0;
}
