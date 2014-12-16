# Common definitions
CC = mpicc #vtcc -vt:cc mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 #-march=corei7 -msse4.1 -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

OBJS = partdiff-par.o askparams.o 

STANDARD_OBJS_DIR = partdiff-par-objs
OPENMP_OBJS_DIR   = partdiff-par-hybrid-objs

STANDARD_OBJS = $(addprefix $(STANDARD_OBJS_DIR)/, $(OBJS))
OPENMP_OBJS   = $(addprefix $(OPENMP_OBJS_DIR)/, $(OBJS))

# Rule to create *.o from *.c
#.c.o:
#	$(CC) -c $(CFLAGS) $*.c

# Targets ...
all: partdiff-par partdiff-par-hybrid

partdiff-par: $(STANDARD_OBJS) Makefile
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ $(STANDARD_OBJS) $(LIBS)

partdiff-par-hybrid: $(OPENMP_OBJS) Makefile
	$(CC) $(CFLAGS) $(LFLAGS) -fopenmp -DOPENMP -o $@ $(OPENMP_OBJS) $(LIBS) #-fopenmp -DOPENMP

clean:
	$(RM) *.o *~ -r $(STANDARD_OBJS_DIR) -r $(OPENMP_OBJS_DIR) -f partdiff-par partdiff-par-hybrid

$(STANDARD_OBJS_DIR)/%.o: %.c
	@mkdir -p $(STANDARD_OBJS_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OPENMP_OBJS_DIR)/%.o: %.c
	@mkdir -p $(OPENMP_OBJS_DIR)
	$(CC) $(CFLAGS) -fopenmp -DOPENMP -c $< -o $@ #-fopenmp -DOPENMP


#partdiff-par.o: partdiff-par.c Makefile
#askparams.o: askparams.c Makefile
#displaymatrix-mpi.o: displaymatrix-mpi.c Makefile
