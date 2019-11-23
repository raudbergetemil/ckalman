# ckalman

Implementation of various control-related algorithms, such as the Kalman filter, LQR, Extended and Cubature Kalman filter in Python and (in the future) C.

## Dependencies 

### Python
- Numpy
- Scipy

### C
- [Meschach](https://github.com/yageek/Meschach)

## Todo
- C implementations 
- Particle filter
- RNN state estimator 
- Feedback linerization for 

## Future 

- Control algorithms such as MPC

## Compilation

- Compile Meschach as shared library
- Compile `kalman.c` as shared library
- Include `kalman.h` in project
- Compile project against `libkalman.so` and `libmescach.so`
