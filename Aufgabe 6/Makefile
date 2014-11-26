# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 #-O0 -g #-ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

OBJS1 = timempi.o
OBJS2 = timempi2.o

# Rule to create *.o from *.c
.c.o:
	$(CC) -c $(CFLAGS) $*.c

# Targets ...
all: timempi timempi2

timempi: $(OBJS1) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS1) $(LIBS)

timempi2: $(OBJS2) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS2) $(LIBS)

clean:
	$(RM) *.o *~ timempi timempi2

timempi.o.o: timempi.c Makefile

timempi2.o.o: timempi2.c Makefile
