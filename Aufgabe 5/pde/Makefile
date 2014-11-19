# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O3 -march=corei7 -pthread #-ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

OBJS = partdiff-posix.o askparams.o

# Rule to create *.o from *.c
.c.o:
	$(CC) -c $(CFLAGS) $*.c

# Targets ...
all: partdiff-posix

partdiff-posix: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	$(RM) *.o *~ partdiff-posix

partdiff-posix.o: partdiff-posix.c Makefile

askparams.o: askparams.c Makefile
