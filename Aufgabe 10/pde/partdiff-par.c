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

#ifdef OPENMP
	#include "omp.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff-par.h"

#define assert(x, str) { if(!x) { printf(str); MPI_Abort(MPI_COMM_WORLD, -1); } }

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
        //das globale N zuweisen, dies wird für die Spalten benutztt
	arguments->globalN = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1; //anzahl der Matrizen ist bei Jacobi immer zwei
	arguments->h = 1.0 / arguments->globalN; //h mittels des globalen N ermitteln

	if(arguments->commSize == 1) //sonderfall für nur ein Prozess betrachten
	{
	  //gesammte Matrix übergeben
	  arguments->start = 0;
	  arguments->end   = arguments->globalN;
	  arguments->N     = arguments->globalN;
	}
        else
	{
#ifdef HALF
	  int k = ((arguments->globalN - 1) * arguments->globalN / 2) / arguments->commSize;
#endif

	  //neuen range berechnet
	  float range = ((float) arguments->globalN) / ((float) arguments->commSize);
	  if(arguments->rank == 0) //für den rank 0 von 0 bis range zuweisen
	  {
	    arguments->start = 0;
#ifndef HALF
	    arguments->end   = range;
#else
	    arguments->end   = 0.5 * (sqrt(8 * k * (arguments->rank + 1) + 1) + 1);
#endif
	  }
	  else if(arguments->rank == arguments->commSize - 1) //für den letzten prozess den rest zu weisen
	  {
#ifndef HALF
	    arguments->start = ((int) range * arguments->rank) - 1;
	    arguments->end   = arguments->globalN;
#else
	    arguments->start   = 0.5 * (sqrt(8 * k * arguments->rank + 1) + 1) - 1;
	    arguments->end   = arguments->globalN;
#endif
	  }
	  else
	  {
#ifndef HALF
	    //allen anderen prozessen etwas dazuwischen zuweisen
	    arguments->start = ((int) range * arguments->rank) - 1;
	    arguments->end   = ((int) range * arguments->rank + range);
#else
	    arguments->start  = 0.5 * (sqrt(8 * k * arguments->rank + 1) + 1) - 1;
	    arguments->end    = 0.5 * (sqrt(8 * k * (arguments->rank + 1) + 1) + 1);
#endif
	  }

	  //die Anzahl der Zeilen berechnen
	  arguments->N = arguments->end - arguments->start;
	}

#ifdef DEBUG
	printf("Rank %i starts at %i to %i\n", arguments->rank, arguments->start, arguments->end);
#endif

	//Standard werte setzen
	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;

	//debug info
	//printf("Rank: %i, go from: %"PRIu64 " to %" PRIu64 " with N: %" PRIu64 " and Global N %"PRIu64 "\n",
	//       arguments->rank, arguments->start, arguments->end, arguments->N, arguments->globalN);
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
        void *p = malloc(size);
	if ((p = malloc(size)) == NULL)
	{
	  printf("Speicherprobleme! (%" PRIu64 " Bytes)\n", size);
	  MPI_Abort(MPI_COMM_WORLD, -1);
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
	//matrix allocieren und zwar der höhe mal der breite
	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (arguments->globalN + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
		        // da die Zeilen globalN lang sind die zuweisung neu berechnen
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
	    //werte für die Seiten zu weisen, dies berechnet sich aus dem start
	    for (i = 0; i < N + 1; i++)
	    {
	      Matrix[g][i][0] = 1.0 - (h * (i + arguments->start));
	      Matrix[g][i][arguments->globalN] = h * (i + arguments->start);
	    }
	  }

	  //da der 0. Prozess die erste Spalte hat hier die Randwerte setzen
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

	  //da der letzte Prozess die letzte Spalte hat, hier die Randwerte setzen
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
calculateJacobi (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options)
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

	//werte voreberechnen
	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

#ifdef OPENMP
	omp_set_dynamic(0);
	omp_set_num_threads(options->number);
#endif

#ifdef DEBUG
	printf("Rank %i compute from %i to %i global %i\n", arguments->rank, arguments->start + 1, arguments->end, arguments->globalN);
#endif

	//iterrieren
	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];

		maxresiduum = 0;

		/* over all rows */
#ifdef OPENMP
		#pragma omp parallel for private(i, j, star, residuum) firstprivate(pih, fpisin, m1, m2, term_iteration) shared(Matrix_Out, Matrix_In, options, arguments) reduction(max:maxresiduum)  default(none)
#endif
		for (i = 1; i < N; i++)
		{
			double fpisin_i = 0.0;
			double posi = i + ((double)arguments->start);
			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)posi);
			}
#ifdef HALF
   		        //berechnen der ganzen Matrix, wie immer
			for (j = 1; j < arguments->start + i; j++) // < i
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

#ifdef HALF
			// == i
			int ii = arguments->start + i;
			star = 0.5 * (Matrix_In[i][ii-1] + Matrix_In[i+1][ii]);
			if (options->inf_func == FUNC_FPISIN)
			{
				star += fpisin_i * sin(pih * (double) ii);
			}
			if (options->termination == TERM_PREC || term_iteration == 1)
			{
				residuum = Matrix_In[i][ii] - star;
				residuum = (residuum < 0) ? -residuum : residuum;
				maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}

			Matrix_Out[i][ii] = star;
#endif
		}

		//wenn wir nicht in rank 0 sind, dann
		if(arguments->rank > 0)
		{
#ifndef HALF
		  //die erste Spalte senden und die berechneten Werte der oberen Prozesses empfangen und in die 0. Spalte eintragen
		  assert(MPI_Sendrecv(Matrix_Out[1], arguments->globalN + 1, MPI_DOUBLE, arguments->rank - 1, arguments->rank,
				      Matrix_Out[0], arguments->globalN + 1, MPI_DOUBLE, arguments->rank - 1, arguments->rank - 1, 
				      MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not send or recive Data");
#else
		  const int send = arguments->start + 1;
		  const int recv = arguments->start + 1;
		  //die erste Spalte senden und die berechneten Werte der oberen Prozesses empfangen und in die 0. Spalte eintragen
		  assert(MPI_Sendrecv(Matrix_Out[1], send, MPI_DOUBLE, arguments->rank - 1, arguments->rank,
				      Matrix_Out[0], recv, MPI_DOUBLE, arguments->rank - 1, arguments->rank - 1, 
				      MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not send or recive Data");
#endif
		}
		if(arguments->rank != arguments->commSize - 1)
		{
#ifndef HALF
		  //gleiches gilt für die untere Spalte
		  assert(MPI_Sendrecv(Matrix_Out[N - 1], arguments->globalN + 1, MPI_DOUBLE, arguments->rank + 1, arguments->rank,
			              Matrix_Out[N], arguments->globalN + 1, MPI_DOUBLE, arguments->rank + 1, arguments->rank + 1, 
				      MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not send or recive Data");
#else
		  const int send = arguments->end;
		  const int recv = arguments->end;
		  //gleiches gilt für die untere Spalte
		  assert(MPI_Sendrecv(Matrix_Out[N - 1], send, MPI_DOUBLE, arguments->rank + 1, arguments->rank,
			              Matrix_Out[N], recv, MPI_DOUBLE, arguments->rank + 1, arguments->rank + 1, 
				      MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not send or recive Data");
#endif
		}

		//das abbruch kriterium prüfen und das maximum an alle senden
		if(options->termination == TERM_PREC || term_iteration == 1)
		{
		  double currentResiduum = maxresiduum;
		  assert(MPI_Allreduce(&maxresiduum, &currentResiduum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not reduce maxresiduum");
		  maxresiduum = currentResiduum;
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* abbrechen wenn das residuum kleiner als die Abbruchbedigung ist */
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
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculateGaussSeidel (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options)
{
	int i, j;                                   /* local variables for loops  */
	double star;                                /* four times center value minus 4 neigh.b values  */
	double residuum;                            /* residuum of current iteration                   */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration  */
	double globalMaxResiduum;                   /* Das globale Maxresiduum der aktuellen Iteration */

	int const N = arguments->N;
	double const h = arguments->h;
	const int rank = arguments->rank;

	double pih = 0.0;
	double fpisin = 0.0;
	int terminate = 0;

	int term_iteration = options->term_iteration;
	//werte voreberechnen
	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	//wenn openmp aktiv ist wichtige einstellungen setzen
#ifdef OPENMP
	omp_set_dynamic(0);
	omp_set_num_threads(options->number);
#endif

	//iterrieren
	while (term_iteration > 0)
	{
	        //Matrix_In und Matrix_Out der uebersichthalber benutzen
		double** Matrix_Out = arguments->Matrix[0];
		double** Matrix_In  = arguments->Matrix[0];
		maxresiduum = 0;
		globalMaxResiduum = 0;

		// Hier wird der erste Rank aussenvor gelassen, da das verfahren sonst blockieren wuerde
		if(arguments->rank > 0)
		{
#ifdef DEBUG
		  printf("Rank %i recv from %i, in iteration %i\n", rank, rank - 1, results->stat_iteration);
#endif
		  //hier werden die werte, welche unten gesendet wurden nun empfangen, hierbei ist rank + iteration noetig, damit
		  //es zu keinem Konfikt zwischen dem globalen maxresiduum kommt.
#ifndef HALF
		  assert(MPI_Recv(Matrix_Out[0], arguments->globalN, MPI_DOUBLE, rank - 1, rank + results->stat_iteration, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the matrix\n");
#else
		  assert(MPI_Recv(Matrix_Out[0], arguments->start + 1, MPI_DOUBLE, rank - 1, rank + results->stat_iteration, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the matrix\n");
#endif
		  if(options->termination == TERM_PREC || term_iteration == 1)
		  {
		    //das globale residuum welches in dieser kommenden iteration vom Prozess rank - 1 berechnet wurde abholen und damit weiter rechnen
		    assert(MPI_Recv(&globalMaxResiduum, 1, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the maxresiduum flag\n");
		  }

		  //das termination flag vom prozess rank - 1 abholen, um zu sehen, ob der Prozess seine letzten iterationen durchfuehren muss
		  if(options->termination == TERM_PREC)
		  {
		    int term = 0;
		    assert(MPI_Recv(&term, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the termination flag\n");
		    //wenn das termination flag noch kleiner ist als jetzt dann den flag erhoehen sonst so lassen,
		    //bsp das Termination flag ist 1 der master Prozess hat aber in der jetzigen iteration beendet also ist term = 2 damit wird auch termination 2 
		    //nicht ueber schrieben, falls vom kleineren Prozess term = 1 oder term = 0 kommen solte.
		    if(terminate < term)
		      terminate = term;
		  }
		}

		//wenn wir uns nicht in der ersten Iteration befinden die veränderten Zeilen tauschen
		if(results->stat_iteration > 0)
		{
		  //da der letzte Prozess keine Zeile nach zu einem nachfolger geschickt hat muss dieser nicht betrachtet werden
		  if(arguments->rank < arguments->commSize - 1)
		  {
#ifdef DEBUG
		    printf("Rank %i recv from %i, in iteration %i\n", rank, rank + 1, results->stat_iteration + 1);
#endif
		    //die werte werten jetzt von den unten geschickten empfangen, hierbei ist rank + iteration + 1 noetig, damit dies nicht
		    //in den Konflikt mit dem gesendeten max residuum kommt.
#ifndef HALF
		    assert(MPI_Recv(Matrix_Out[N], arguments->globalN, MPI_DOUBLE, rank + 1, rank + results->stat_iteration + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the matrix\n");
#else
		    assert(MPI_Recv(Matrix_Out[N], arguments->end + 1, MPI_DOUBLE, rank + 1, rank + results->stat_iteration + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the matrix\n");
#endif

		    //das termination flag auch von oben abholen, damit dies wenn es auf 1 gesetzt wurde auch den kommenden prozessen dies mitteilt
		    if(options->termination == TERM_PREC)
		    {
		      int term = 0;
		      assert(MPI_Recv(&term, 1, MPI_INT, rank + 1, rank + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the termination flag\n");
		      //wenn das termination flag noch kleiner ist als jetzt dann den flag erhoehen sonst so lassen,
		      //bsp das Termination flag ist 1 der master Prozess hat aber in der jetzigen iteration beendet also ist term = 2 damit wird auch termination 2 
		      //nicht ueber schrieben, falls vom kleineren Prozess term = 1 oder term = 0 kommen solte.
		      if(terminate < term)
			terminate = term;
		    }
		  }
		}
#ifdef OPENMP //DIE OPENMP Implementierung
		//wenn mehr als ein thread existiert eine red-back gauss seidel methode fahren
		if(options->number > 1)
		{
		  //alternativ mittels openmp allerdings wird hier im gegensatz zur mpi implementierung red black eingesetzt.
                  #pragma omp parallel for private(i, j, star) reduction(max:maxresiduum)
		  for (i = 1; i < N; i++)
		  {
		    double fpisin_i = 0.0;
		    double posi = i + ((double)arguments->start);
		    if (options->inf_func == FUNC_FPISIN)
		    {
		      fpisin_i = fpisin * sin(pih * (double)posi);
		    }

		    //berechnen der ganzen Matrix, wie immer
		    for (j = i % 2; j < (int) arguments->globalN; j+=2)
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
		  }

                  #pragma omp parallel for private(i, j, star) reduction(max:maxresiduum)
		  for (i = 1; i < N; i++)
		  {
		    double fpisin_i = 0.0;
		    double posi = i + ((double)arguments->start);
		    if (options->inf_func == FUNC_FPISIN)
		    {
		      fpisin_i = fpisin * sin(pih * (double)posi);
		    }
		    //berechnen der ganzen Matrix, wie immer
		    for (j = i % 2 + 1; j < (int) arguments->globalN; j+=2)
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
		  }
		}
	        else //wenn nur ein thread existiert kein red - black gausseidel fuer die openmp implementierung
		{
		  for (i = 1; i < N; i++)
		  {
		    double fpisin_i = 0.0;
		    double posi = i + ((double)arguments->start);
		    if (options->inf_func == FUNC_FPISIN)
		    {
		      fpisin_i = fpisin * sin(pih * (double)posi);
		    }
		    //berechnen der ganzen Matrix, wie immer
		    for (j = 1; j < (int) arguments->globalN; j++)
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
		  }
		}
#else //DIE NICHT OPENMP Methode (Ohne red - black)
		//die eigentliche berechnung erfolgt wie immer.
		for (i = 1; i < N; i++)
		{
			double fpisin_i = 0.0;
			double posi = i + ((double)arguments->start);
			if (options->inf_func == FUNC_FPISIN)
			{
			  fpisin_i = fpisin * sin(pih * (double)posi);
			}
#ifdef HALF
			for (j = 1; j < (int) (arguments->start + i); j++)
#else
			 //berechnen der ganzen Matrix, wie immer
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
#ifdef HALF
			int ii = arguments->start + i;
			star = 0.5 * (Matrix_In[i][ii-1] + Matrix_In[i+1][ii]);
			if (options->inf_func == FUNC_FPISIN)
			{
				star += fpisin_i * sin(pih * (double) ii);
			}
			if (options->termination == TERM_PREC || term_iteration == 1)
			{
				residuum = Matrix_In[i][ii] - star;
				residuum = (residuum < 0) ? -residuum : residuum;
				maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}
			Matrix_Out[i][ii] = star;
#endif

		}
#endif
		//das maxresiduum der Iteration checken und mit dem bisherigen vergleichen
		maxresiduum = (maxresiduum < globalMaxResiduum) ? globalMaxResiduum : maxresiduum;

		if(arguments->commSize > 1)
		{
		  //da der 0. Prozess keinen vorgaenger hat gilt dies nur fuer die Prozesse von 1, ..., k
		  //auserdem nicht durch fuehren, wenn die prozesse beenden, da sonst werte gesendet werden, die nicht mehr benoetigt werden
		  //und es somit zu einer falschen ausgabe kommt.
		  if(arguments->rank > 0 && term_iteration > 1 && terminate != 2)
		  {
#ifdef DEBUG
		    printf("Rank %i send to %i, in iteration %i\n", rank, rank - 1, results->stat_iteration + 1);
#endif
#ifndef HALF
		    //Die obere Zeile an den uebergeordenten Prozess senden, da dieser die werte fuer die letzte Zeile benoetigt.
		    assert(MPI_Send(Matrix_Out[1], arguments->globalN, MPI_DOUBLE, rank - 1, rank + results->stat_iteration + 1, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the matrix part\n");
#else
		    assert(MPI_Send(Matrix_Out[1], arguments->start + 2, MPI_DOUBLE, rank - 1, rank + results->stat_iteration + 1, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the matrix part\n");
#endif
		    //das termination flag an den kommenden prozess weiter reichen
		    if(options->termination == TERM_PREC)
		    {
		      assert(MPI_Send(&terminate, 1, MPI_INT, rank - 1, rank, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the termination flag\n");
		    }
		  }
		  //da der letzte Prozess keinen nachfolger hat, nur fuer die Prozesse 0, ..., k - 1 durchfuehren
		  if(arguments->rank != arguments->commSize - 1)
		  {
#ifdef DEBUG
		    printf("Rank %i send to %i, in iteration %i\n", rank, rank + 1, results->stat_iteration + 1);
#endif
		    //Die unteren Matrix werte sind fuer den naechsten Prozess wichtig, da dieser diese fuer die naechste Iteration benoetigt.
		    //Daher senden wir diese werte an den Prozess plus 1. Der Tag ist außerdem wichtig, damit der Prozess die richtigen werte spaeter
		    //abfragt, damit keine falschen werte entstehen.
#ifndef HALF
		    assert(MPI_Send(Matrix_Out[arguments->N - 1], arguments->globalN, MPI_DOUBLE, rank + 1, rank + results->stat_iteration + 1, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the matrix part\n");
#else
		    assert(MPI_Send(Matrix_Out[arguments->N - 1], arguments->end, MPI_DOUBLE, rank + 1, rank + results->stat_iteration + 1, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the matrix part\n");
#endif
		    //das neuberechnete residuum an den naechsten Prozess weiter reichen
		    if(options->termination == TERM_PREC || term_iteration == 1)
		    {
		      assert(MPI_Send(&maxresiduum, 1, MPI_DOUBLE, rank + 1, rank, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the maxresiduum\n");
		      //das termination flag an den kommenden prozess weiter reichen
		      if(options->termination == TERM_PREC)
		      {
			assert(MPI_Send(&terminate, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the termination flag\n");
		      }
		    }
		  }
		}

		//vorwarts iterieren
		results->stat_iteration++;
		//das residuum setzten
		results->stat_precision = maxresiduum;
		if(arguments->commSize > 1)
		{
		  if(terminate == 2)
		  {
#ifdef DEBUG
		    printf("Rank %i exit now\n", rank);
#endif
		    term_iteration = 0; //dieser Prozess befindet sich nun in der selben iteration wie der Prozess mit dem rank 0
		  }
		  if(rank == 0 && terminate == 1)
		  {
#ifdef DEBUG
		    printf("Rank 0 recv. termination flag\n");
#endif
		    terminate = 2; // das termination flag auf 2 sezten, dies bedeutet fuer alle Prozesse, dass dies die letzte iteration der matrix ist.
	          }
		}
		/* abbrechen wenn das residuum kleiner als die Abbruchbedigung ist */
		if (options->termination == TERM_PREC)
		{
		  if (maxresiduum < options->term_precision)
		  {
		    if(arguments->commSize == 1)
		    {
			term_iteration = 0;
		    }
		    else
		    {
		      terminate = (terminate < 1) ? 1 : terminate; //dieser Prozess hat das abbruch kriterium gefunden, damit wird dieses nun an den Prozess 0 herrangetragen und dieser beendet das program
		    }
		  }
		}
		else if (options->termination == TERM_ITER)
		{
		  term_iteration--;
		}
	}

	//da das akutelle maxresiduum der Iteration immer weiter nach unten an den naechsten rank verschickt wurde jetzt das residuum an den
	//ersten prozess zurueck senden damit dies korrekt ausgegeben werden kann.
	//Die barrier ist noetig, damit alle Prozesse hier auf dem gleichen stand sind und das korrekte residuum zurueck gegeben wird.
	if(arguments->commSize > 1)
	{
	  assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "Could not wait for all processes\n");
	  if(rank == 0) //empfangen
	  {
	    assert(MPI_Recv(&maxresiduum, 1, MPI_DOUBLE, arguments->commSize - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv the maxresiduum from the last process\n");
	  }
	  else if(rank == arguments->commSize - 1) //senden
	  {
	    assert(MPI_Send(&maxresiduum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send the maxresiduum to the first process\n");
	  }
	  assert(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "Could not wait for all processes\n"); //warten
#ifdef DEBUG
	  printf("Done %i iterations\n", results->stat_iteration);
#endif


          maxresiduum = (maxresiduum < globalMaxResiduum) ? globalMaxResiduum : maxresiduum;
	  assert(MPI_Allreduce(&maxresiduum, &globalMaxResiduum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not reduce all");
	  maxresiduum = globalMaxResiduum;
	}

	//da das residuum fuer die restlichen prozesse egal ist kann hier einfach zu gewiesen werden, da Prozess 0 das korrekte residuum hat.
        results->stat_precision = maxresiduum;
	results->m = 0; // da wir ausserdem im Gauss-Seidel verfahren sind ex. nur eine matrix, damit ist m immer 0
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
#ifdef OPENMP
	  if(options->number > 1)
	    printf("Red-Black Gauss-Seidel\n\nWARNIG RED-BLACK GAUSS SEIDEL PRODUCE NOT CORRECT RESULTS FOR SMALL ITERATIONS AND MATRICES BECAUSE OF THE MATHEMATICAL BACKGROUND\n");
	  else
	    printf("Gauss-Seidel");
#else
	  printf("Gauss-Seidel");
#endif
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
}

#ifdef DEBUG
static
void
DebugDisplayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	int x, y;

	double** Matrix = arguments->Matrix[results->m];
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
#endif

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
    printf("\nMatrix:\n");

#ifdef HALF
  double* smallMatrix = malloc(sizeof(double) * 9 * 9);
#endif

  //printf("rank %i from %i to %i\n", rank, from, to);
  for (y = 0; y < 9; y++)
  {
    int line = y * (options->interlines + 1);

    if (rank == 0)
    {
      /* check whether this line belongs to rank 0 */
      if (line < from || line > to)
      {
	//debug ausgabe
#ifdef DEBUG
	if(rank == 0)
	  printf("Need line: %i, y=%i\n", line, y);
#endif
        /* use the tag to receive the lines in the correct order
         * the line is stored in Matrix[0], because we do not need it anymore */
        assert(MPI_Recv(Matrix[0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status) == MPI_SUCCESS, "Could not recive output data");
      }
    }
    else
    {
      if (line >= from && line <= to)
      {
	//debug ausgabe
#ifdef DEBUG
	printf("Send line %i, rank %i, y=%i\n", line, arguments->rank, y);
#endif
        /* if the line belongs to this process, send it to rank 0
         * (line - from + 1) is used to calculate the correct local address */
        assert(MPI_Send(Matrix[line - from + 1], elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send output data");
      }
    }

    if (rank == 0)
    {
      //debug ausgabe
      //if(rank == 0)
      //  printf("Print line: %i, y=%i\n", line, y);
      for (x = 0; x < 9; x++)
      {
        int col = x * (options->interlines + 1);
        if (line >= from && line <= to)
        {
#ifndef HALF
          /* this line belongs to rank 0 */
          printf("%7.4f", Matrix[line][col]);
#else
	  smallMatrix[y * 9 + x] = Matrix[line][col];
#endif
        }
        else
        {
#ifndef HALF
          /* this line belongs to another rank and was received above */
          printf("%7.4f", Matrix[0][col]);
#else
  	  smallMatrix[y * 9 + x] = Matrix[0][col];
#endif
        }
      }
#ifndef HALF
	printf("\n");
#endif
    }
  }

#ifdef HALF
  if(rank == 0)
  {
    for(int i = 1; i < 9; i++)
    {
      for(int j = 1; j < i; j++)
      {
	smallMatrix[j * 9 + i] = smallMatrix[i * 9 + j];
      }
    }

    for(int i = 0; i < 9; i++)
    {
      for(int j = 0; j < 9; j++)
      {
	printf("%7.4f", smallMatrix[i * 9 + j]);
      }
      printf("\n");
    }
    free(smallMatrix);
  }
#endif
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

	//initialisieren
	assert(MPI_Init(&argc, &argv) == MPI_SUCCESS, "Could not init mpi\n");
	//rank herrausfinden
	assert(MPI_Comm_rank(MPI_COMM_WORLD, &arguments.rank) == MPI_SUCCESS, "Could not get the rank of this process\n");
	//anzahl der Prozesse herrausfinden
	assert(MPI_Comm_size(MPI_COMM_WORLD, &arguments.commSize) == MPI_SUCCESS, "Could not get the comm size of processes\n");

	/* get parameters */
	AskParams(&options, argc, argv, arguments.rank);
	//wenn mehr als doppelt so viele prozesse für die interlines vorhanden sind abbrechen, da sich prozesse sonst Zeilen teilen und es zu bugs kommt.
	if(options.interlines * 8 + 9 - 1 < arguments.commSize * 2)
	{
	  //prüfen ob mehr prozesse vorhanden sind als die Matrix gross ist
	  if(arguments.rank == 0)
	    printf("\n\n\n\tToo many processes for this small matrix, waste of resources and bugs can appear!\n\tPlease restart with a fewer count of processes.\n\tExit now.\n\n\n\n");

	  fflush(stdout);
	  //beenden
	  MPI_Abort(MPI_COMM_WORLD, - 1);
	}

	initVariables(&arguments, &results, &options);
	allocateMatrices(&arguments);
	initMatrices(&arguments, &options);

	gettimeofday(&start_time, NULL);
	if(options.method == METH_JACOBI)
	  calculateJacobi(&arguments, &results, &options);
	else
	  calculateGaussSeidel(&arguments, &results, &options);
	gettimeofday(&comp_time, NULL);
	MPI_Barrier(MPI_COMM_WORLD);

	//statistik ausgeben
	if(arguments.rank == 0)
	{
	  displayStatistics(&arguments, &results, &options);
	}
	//die matrix um die erweiterten elemente verkleinern
	int add = (arguments.rank > 0) ? 1 : 0;
	int sub = (arguments.rank < arguments.commSize) ? 1 : 0;
	//ausgeben
	DisplayMatrix(&arguments, &results, &options, arguments.rank, arguments.commSize, arguments.start + add, arguments.end - sub);
	//daten freigeben
	freeMatrices(&arguments);

	//beenden
	//printf("Process: %i exit now\n", arguments.rank);
	assert(MPI_Finalize() == MPI_SUCCESS, "Could not finalize mpi\n");
	return 0;
}
