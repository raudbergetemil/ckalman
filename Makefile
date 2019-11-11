
MESCHACH_PATH:=/home/eml/Downloads/Meschach

all: kalman

kalman: kalman.c 
	gcc -I$(MESCHACH_PATH) -o libkalman.so -lm -L$(MESCHACH_PATH)/libmeschach.so -fPIC -shared kalman.c 