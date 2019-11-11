# ckalman

State estimation toolbox. 

## Dependencies 

### Python
- Numpy

### C
- [Meschach](https://github.com/yageek/Meschach)

## Todo
- C implementations 
- Particle filter
- RNN state estimator 

## Future 

- Control algorithms such as LQR, MPC

## Compilation

- Compile Meschach as shared library
- Compile `kalman.c` as shared library
- Include `kalman.h` in project
- Compile project against `libkalman.so` and `libmescach.so`
