# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -march=corei7 -msse4.1 -O3 #-O2 -ggdb -pg -g
LFLAGS = $(CFLAGS)
LIBS   = -lm
OMP    = -fopenmp -DOPENMP -DSCHEDULE_TYPE=dynamic -DSCHEDULE_SIZE=4

STANDARD_OBJS_DIR   = partdiff-seq-objs
MP_BASE_OBJS_DIR    = partdiff-openmp-objs
MP_ROW_OBJS_DIR     = partdiff-openmp-spalten-objs
MP_COL_OBJS_DIR     = partdiff-openmp-zeilen-objs
MP_ELEM_OBJS_DIR    = partdiff-openmp-element-objs

OBJS = partdiff-seq.o askparams.o
STANDARD_OBJS = $(addprefix $(STANDARD_OBJS_DIR)/, $(OBJS))
MP_BASE_OBJS  = $(addprefix $(MP_BASE_OBJS_DIR)/, $(OBJS))
MP_ROW_OBJS   = $(addprefix $(MP_ROW_OBJS_DIR)/, $(OBJS))
MP_COL_OBJS   = $(addprefix $(MP_COL_OBJS_DIR)/, $(OBJS))
MP_ELEM_OBJS  = $(addprefix $(MP_ELEM_OBJS_DIR)/, $(OBJS))

# Targets ...
all: partdiff-seq partdiff-openmp partdiff-openmp-spalten partdiff-openmp-zeilen partdiff-openmp-element

partdiff-seq: $(STANDARD_OBJS)
	$(CC) $(LFLAGS) -o $@ $(STANDARD_OBJS) $(LIBS)

partdiff-openmp: $(MP_BASE_OBJS)
	$(CC) $(LFLAGS) $(OMP) -o $@ $(MP_BASE_OBJS) $(LIBS)

partdiff-openmp-spalten: $(MP_ROW_OBJS)
	$(CC) $(LFLAGS) $(OMP) -o $@ $(MP_ROW_OBJS) $(LIBS)	

partdiff-openmp-zeilen: $(MP_COL_OBJS)
	$(CC) $(LFLAGS) $(OMP) -o $@ $(MP_COL_OBJS) $(LIBS)

partdiff-openmp-element: $(MP_ELEM_OBJS)
	$(CC) $(LFLAGS) $(OMP) -o $@ $(MP_ELEM_OBJS) $(LIBS)

$(STANDARD_OBJS_DIR)/%.o: %.c
	@mkdir -p $(STANDARD_OBJS_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(MP_BASE_OBJS_DIR)/%.o: %.c
	@mkdir -p $(MP_BASE_OBJS_DIR)
	$(CC) $(CFLAGS) $(OMP) -DOVER_BASE -c $< -o $@

$(MP_ROW_OBJS_DIR)/%.o: %.c
	@mkdir -p $(MP_ROW_OBJS_DIR)
	$(CC) $(CFLAGS) $(OMP) -DOVER_ROW -c $< -o $@

$(MP_COL_OBJS_DIR)/%.o: %.c
	@mkdir -p $(MP_COL_OBJS_DIR)
	$(CC) $(CFLAGS) $(OMP) -DOVER_COL -c $< -o $@	

$(MP_ELEM_OBJS_DIR)/%.o: %.c
	@mkdir -p $(MP_ELEM_OBJS_DIR)
	$(CC) $(CFLAGS) $(OMP) -DOVER_ELEM -c $< -o $@	

clean:
	$(RM) *.o *~ -r $(STANDARD_OBJS_DIR) -r $(MP_BASE_OBJS_DIR) -r $(MP_ROW_OBJS_DIR) -r $(MP_COL_OBJS_DIR) -r $(MP_ELEM_OBJS_DIR) partdiff-seq partdiff-openmp partdiff-openmp-spalten partdiff-openmp-zeilen partdiff-openmp-element
