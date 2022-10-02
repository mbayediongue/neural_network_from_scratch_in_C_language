CC=gcc
CFLAGS=-Wall -ansi -pedantic 
LDFLAGS=
EXEC=main_executable

all: $(EXEC)

main_executable: neurone.o neurone_training.o main.o
		$(CC) -o $@ $^ $(LDFLAGS) -lm

neurone.o: neurone.c
		$(CC) -o $@ -c $< $(CFLAGS)

neurone_training.o: neurone_training.c neurone.h
		$(CC) -o $@ -c $< $(CFLAGS)

main.o: main.c neurone.h mlp.h neurone_training.h
		$(CC) -o $@ -c $< $(CFLAGS)

clean:
		rm -f *.o core

mrproper: clean
		rm -f $(EXEC)



