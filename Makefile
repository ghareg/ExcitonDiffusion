all: prog

prog: main.o operator.o
	nvcc -std=c++11 -o VecCalc -L/share/usr/compilers/cuda/8.0.44/lib64 -lcudart -lcufft -L/share/usr/gsl/1.16_intel/lib -lgsl -lgslcblas main.o operator.o
	rm -rf *.o

test: phase_oscillator_ensemble.o
	nvcc -std=c++11 -o program -L/share/usr/compilers/cuda/8.0.44/lib64 -lcudart -lcufft phase_oscillator_ensemble.o
	rm -rf *.o

main.o:
	nvcc -c -std=c++11 -O3 -arch=sm_35 main.cu

operator.o:
	nvcc -c -std=c++11 -O3 -arch=sm_35 operator.cu

phase_oscillator_ensemble.o:
	nvcc -c -std=c++11 -arch=sm_35 phase_oscillator_ensemble.cu

clean: rm -rf *o VecCalc
