#ifndef OPERATOR_H_
#define OPERATOR_H_
#include "constants.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <cufft.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>


typedef int Count;
typedef float dtype;
typedef thrust::device_vector<dtype> sType;
typedef thrust::host_vector<dtype> shType;

struct Compl
{
	dtype real;
	dtype imag;
};

typedef thrust::device_vector<Compl> cType;
typedef thrust::device_ptr<Compl> pType;
typedef thrust::device_ptr<const Compl> cpType;


class polaritonSchrodinger
{
public:
	polaritonSchrodinger(Count xSize, Count ySize, int rank, int batch);

	~polaritonSchrodinger();

	void operator()(const sType& x, sType& dxdt, dtype t);

	struct KSysFunctor {
		KSysFunctor();

		template<class Tuple>
        __host__ __device__
        void operator()(Tuple t);
	};

	struct XYSysFunctor {
		float CXLft_;

		XYSysFunctor(float CXLft);

		template<class Tuple>
        __host__ __device__
        void operator()(Tuple t);
	};

	struct MultFunctor {
		dtype mult_;

		MultFunctor(dtype mult);
		
		template<class Type>
        __host__ __device__
        Type operator()(const Type& x);
	};
	
	struct SumMultFunctor {
		dtype mult_;

		SumMultFunctor(dtype mult);
		
		template<class Type>
        __host__ __device__
        Type operator()(const Type& x, const Type& y);
	};
	
	void initUScat();

private:
	float PolaritonDispersion(float k, int p, int s);
	void createMat();

private:
	Count xSize_;
	Count ySize_;
	Count Size_;
	dtype Sizem1_;
	int rank_;
	int batch_;
	sType xT_;
	Count* n_;
	sType tex_;
	cType Sex_;
	cType pump_;
	sType UScat_;
	cufftHandle plan_;
	shType cr_;
	shType UScatl_;
	thrust::minstd_rand rng_;
	thrust::random::normal_distribution<dtype> dist_;
};

class observer
{
public:
	observer(Count xSize, Count ySize, shType& xInt, dtype tmax, dtype tdif);
	~observer();

	void initialize();
	void operator() (const sType& x, dtype t);
	void normalize();

	struct  OverFunctor {

		OverFunctor(dtype sqrt2);

		dtype sqrt2_;
		
		template<class Tuple>
        __host__ __device__
        void operator()(Tuple t);
	};

private:
	Count xSize_;
	Count ySize_;
	shType& xInt_;
	shType xTemp_;
//	sType kTemp_;
	sType xDTemp_;
	dtype tmax_;
	dtype tdif_;
	Count steps_;
	thrust::plus<dtype> op_;
//	cufftHandle plan_;
};




#endif
