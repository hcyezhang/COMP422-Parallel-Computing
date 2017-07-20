CC=g++
CFLAGS=-O2

all: worksharing-serial worksharing-parallel nonworksharing-parallel 

worksharing-serial: 
	$(CC) $(CFLAGS) -o lu-worksharing-serial lu-worksharing-serial.cpp

worksharing-parallel:
	$(CC) $(CFLAGS) -fopenmp -o lu-worksharing-parallel lu-worksharing-parallel.cpp
nonworksharing-parallel:
	$(CC) $(CFLAGS) -fopenmp -o lu-non-worksharing lu-non-worksharing.cpp 

clean:
	rm lu-worksharing-serial lu-worksharing-parallel lu-non-worksharing
