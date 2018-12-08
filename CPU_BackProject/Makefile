C        	= gcc
GCC_FLAGS  	= -O3 
LD_FLAGS    	= -lm
EXE	        = main
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.c support.h
	$(C) -c -o $@ main.c $(GCC_FLAGS)

support.o: support.c support.h
	$(C) -c -o $@ support.c $(GCC_FLAGS)

$(EXE): $(OBJ)
	$(C) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
