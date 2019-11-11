#include <stdio.h>
#include <matrix.h>
#include <matrix2.h>
#include <matlab.h>

#define N_STATES 2
#define N_INPUTS 2
#define N_MEASUREMENTS 2

void kalman_prediction();
void kalman_update();