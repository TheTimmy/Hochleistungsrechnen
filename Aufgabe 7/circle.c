#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <mpi.h>

//small error check / does the same as assert but close the process by MPI_Abort
#define checkError(x, str) if(!x) { printf(str); MPI_Abort(MPI_COMM_WORLD, -1); }

/*
 * init(N, max, rank, count)
 * N    , int, real size of the array. Marks how many elements should be randomly created
 * max  , int, the size of the array
 * rank , int, the rank of the current process
 * count, int, the count of all processes in the comm world 
 * returns a randomized array with the size of max
 */
int* init (int N, int max, const int rank, const int count)
{
  //allocate for the biggest buffer and set values to -1
  int* buf = malloc(sizeof(int) * max);
  //error check
  checkError(!(buf == NULL), "Could not allocate enough memory\n");

  srand(time(NULL) + rank * count); //erzeugt unterschiedliche werte auf unterschiedlichen Prozessen
  for (int i = 0; i < N; i++)
  {
    buf[i] = rand() % 25; //do not modify %25
  }
  //mark all elements as unnessesary
  for(int i = N; i < max; i++)
  {
    buf[i] = -1;
  }

  return buf;
}

/*
 * circle(buf, N, rank, procCount)
 * buf      , int*, the buffer with the values of the proccess which should be passed to the next process
 * N        , int , the size of the buffer
 * rank     , int , the rank of the current process
 * procCount, int, the number of all processes
 * returns the buffer which holds the same first element as the buffer of the process with rank 0
 */
int* circle (int* buf, const int N, const int rank, const int procCount)
{
  //firstElement is the first elment of the buffer from process 0
  int firstElement = 0, stop = 0, quit = 0;
  //allocate some new space
  int* recvBuf = malloc(sizeof(int) * N);
  checkError(!(recvBuf == NULL), "Could not allocate enought memory\n");
  memset(recvBuf, -1, N);

  int from = rank - 1;
  int to = (rank + 1) % procCount;
  if(from == -1) { from = procCount - 1; }

  if(rank == 0) { firstElement = buf[0]; }
  //send the first element to all other processes
  checkError(MPI_Bcast(&firstElement, 1, MPI_INT, 0, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not broadcast the first Element\n");
  for(int i = 0; i < procCount - 1; i++)
  {
    checkError(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "Could not wait for other processes");
    if(stop == 1) { break; }
    //recive data from rank - 1 and send current buffer to rank + 1
    checkError(MPI_Sendrecv(buf,N,MPI_INT,to,0,recvBuf,N,MPI_INT,from,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE) == MPI_SUCCESS,"Could not send or recive Data\n");
    //checkError(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS, "Could not wait for other processes");
    //copy the buffers
    for(int j = 0; j < N; j++)
    {
      buf[j] = recvBuf[j];
    } 
    
    //check for exit
    //check if the last process found the first value of process 0
    if(rank == procCount - 1 && buf[0] == firstElement) { stop = 1; }
    //share the stop value with the other processes
    checkError(MPI_Allreduce(&stop, &quit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not check for exit\n");
    stop = quit;
  }

  //free the allocated buffer
  free(recvBuf);
  return buf;
}

/*
 * getArraySize(argv)
 * argv, char*, the arguments from the terminal
 * returns the size of the whole array
 */
int getArraySize(char* argv)
{
  char arg[256];
  sscanf(argv, "%s", arg);
  return atoi(arg);
}

/*
 * recvArraySize(size, rank, procCount)
 * size,      int, size of the whole array which should be splitted
 * rank,      int, rank of the process
 * procCount, int, number of processes
 * returns the size of the sub array and the maximum size in a pointer  
 */
int* recvArraySize(const int size, const int rank, const int procCount)
{
  //data is smaller than proccess count => error
  if(size < procCount)
  {
    return NULL;
  }

  int* count = malloc(sizeof(int) * 2);
  memset(count, 0, 2);
  if(rank == 0)
  {
    //create step size for all processes
    int* stepData = malloc(sizeof(int) * procCount);
    checkError(!(stepData == NULL), "Could not allocate enough memory\n");

    //calculate the step size
    int all = 0;
    int step = size / procCount;
    int max = step;
    for(int i = 0; i < procCount - 1; i++)
    {
      all += step;
      stepData[i] = step;
    }

    //give all the rest the last process
    int rest = size - all;
    max = (max < rest) ? rest : max;
    stepData[procCount - 1] = rest;
    count[0] = stepData[0];
    count[1] = max;

    //send the step size to every process which belongs to
    for(int i = 1; i < procCount; i++)
    {
      int sendData[2] = {stepData[i], max};
      checkError(MPI_Send(&sendData, 2, MPI_INT, i, 0, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send Data\n");
    }
    //free temp data
    free(stepData);
  }
  else
  {
    MPI_Recv(count, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  //Debug Data
  //printf("rank %i size %i max %i\n", rank, count[0], count[1]);
  //fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  return count;
}

/*
 * printAllBuffers(str, buf, max, rank, procCount)
 * str      , char*, a small string which should be printed before the elments get printed
 * buf      , int* , the buffer with the elements to print
 * max      , int  , the size of the buffer
 * rank     , int  , the rank of the current process
 * procCount, int, the count of all processes 
 * returns nothing
 */
void printAllBuffers(const char* str, int* buf, const int max, const int rank, const int procCount)
{
  if(rank == 0)
  {
    //print own data
    printf("%s", str);
    for (int i = 0; i < max; i++)
    {
      if(buf[i] == -1) { break; }
      printf ("rank %d: %d\n", 0, buf[i]);
    }

    //print data from processes
    int* outBuf = malloc(sizeof(int) * max);
    checkError(!(outBuf == NULL), "Could not allocate enough memory\n");
    for(int i = 1; i < procCount; i++)
    {
      //recive the data to print from the process
      checkError(MPI_Recv(outBuf, max, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS, "Could not recv Data\n");
      printf("%s", str);
      for (int j = 0; j < max; j++)
      {
	//dont print unnessesary data
	if(outBuf[j] == -1) { break; }
	
	printf ("rank %d: %d\n", i, outBuf[j]);
	fflush(stdout);
      }
    }

    //free allocated data
    free(outBuf);
  }
  else
  {
    //send the data to print to the root
    checkError(MPI_Send(buf, max, MPI_INT, 0, 0, MPI_COMM_WORLD) == MPI_SUCCESS, "Could not send Data\n");    
  }

  //wait for all to finish
  MPI_Barrier(MPI_COMM_WORLD);
}

/*
  The main function, does what it does, so no comment here ;)
 */
int main (int argc, char** argv)
{
  int* buf;
  if (argc > 2)
  {
    printf("Arguments error\n");
    return EXIT_FAILURE;
  }

  //get the size of the whole array
  const int dataSize = getArraySize(argv[1]);
  if(dataSize == 0) 
  { 
    printf("No elements to process. Exit now!\n");
    return EXIT_FAILURE;
  }

  //init mpi and check the return for errors
  checkError(MPI_Init(&argc, &argv) == MPI_SUCCESS, "Could not init MPI\n");
 
  //get the rank of the process
  int rank = 0;
  checkError(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS, "Could not get the rank\n");
  //get the number of all processes and quit if there are only one process
  int procCount = 0;
  checkError(MPI_Comm_size(MPI_COMM_WORLD, &procCount) == MPI_SUCCESS, "Coult not get the comm size\n");
  checkError(procCount == 1, "Not enought processes to proceed. Exit now!\n");

  int* range = recvArraySize(dataSize, rank, procCount);  
  checkError(!(range == NULL), "There are more processes then data. Exit now!\n");
  const int N = range[0]; //data in this buffer
  const int max = range[1] + 1; // the maximal buffer size
  
  //init a random buffer
  buf = init(N, max, rank, procCount); //init buffer of size max + 1
  printAllBuffers("\nBEFORE\n", buf, max, rank, procCount); //print all data

  //circle this buffer
  circle(buf, max, rank, procCount); //circle all buffers
  MPI_Barrier(MPI_COMM_WORLD);

  //print new data
  printAllBuffers("\nAFTER\n", buf, max, rank, procCount);

  //free the buffer
  free(buf);
  free(range);

  //exit
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
