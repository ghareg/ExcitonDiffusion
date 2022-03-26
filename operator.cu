#include "operator.h"
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thrust/iterator/constant_iterator.h>

polaritonSchrodinger::polaritonSchrodinger(Count xSize, Count ySize, int rank, int batch): xSize_(xSize), ySize_(ySize), Size_(xSize_ * ySize_), Sizem1_(1.0f / Size_), rank_(rank), batch_(batch), xT_(2 * batch_ * Size_), n_(new Count[rank_]), tex_(), Sex_(), pump_(), UScat_(Size_), cr_(Size_), UScatl_(Size_), rng_(), dist_(0.0f, 1.0f){
	n_[0] = xSize;
	n_[1] = ySize;
	createMat();
	cufftPlanMany(&plan_, rank_, n_, NULL, 1, Size_, NULL, 1, Size_, CUFFT_C2C, batch_);
}

void polaritonSchrodinger::operator() (const sType& x, sType& dxdt, dtype t)
{
	cufftComplex* xc = reinterpret_cast<cufftComplex*>(const_cast<dtype*>(thrust::raw_pointer_cast(x.data())));
	cufftComplex* xTc_ = reinterpret_cast<cufftComplex*>(const_cast<dtype*>(thrust::raw_pointer_cast(xT_.data())));
	cufftExecC2C(plan_, xc,  xTc_, CUFFT_FORWARD);

	pType phi1It(reinterpret_cast<Compl*>(thrust::raw_pointer_cast(xT_.data())));
	pType phi2It = phi1It + Size_;
	pType xTend  = phi2It + Size_;

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(phi1It, phi2It, tex_.begin(),
					Sex_.begin(), phi1It, phi2It)), 
			thrust::make_zip_iterator(thrust::make_tuple(phi2It, xTend, tex_.end(),
					Sex_.end(), phi2It, xTend)), KSysFunctor());

	cufftComplex* dxdtc = reinterpret_cast<cufftComplex*>(const_cast<dtype*>(thrust::raw_pointer_cast(dxdt.data())));

	cufftExecC2C(plan_, xTc_, dxdtc, CUFFT_INVERSE);

	cpType cphi1It(reinterpret_cast<const Compl*>(thrust::raw_pointer_cast(x.data())));
	cpType cphi2It = cphi1It + Size_;
	cpType xend  = cphi2It + Size_;

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cphi1It, cphi2It, pump_.begin(), UScat_.begin(),
					phi1It, phi2It)),
			thrust::make_zip_iterator(thrust::make_tuple(cphi2It, xend, pump_.end(), UScat_.end(),
					phi2It, xTend)), XYSysFunctor(CXLft));
	
	thrust::transform(xT_.begin(), xT_.end(), dxdt.begin(), dxdt.begin(), SumMultFunctor(Sizem1_));
}


polaritonSchrodinger::KSysFunctor::KSysFunctor()
{
}

template<class Tuple>
 __host__ __device__
void polaritonSchrodinger::KSysFunctor::operator()(Tuple t)
{
	Compl p1;
	Compl p2;
	p1.real = thrust::get<2>(t) * thrust::get<0>(t).imag + thrust::get<3>(t).real * thrust::get<1>(t).imag + thrust::get<3>(t).imag * thrust::get<1>(t).real;
	p1.imag = -thrust::get<2>(t) * thrust::get<0>(t).real - thrust::get<3>(t).real * thrust::get<1>(t).real + thrust::get<3>(t).imag * thrust::get<1>(t).imag;
	p2.real = thrust::get<2>(t) * thrust::get<1>(t).imag + thrust::get<3>(t).real * thrust::get<0>(t).imag - thrust::get<3>(t).imag * thrust::get<0>(t).real;
	p2.imag = -thrust::get<2>(t) * thrust::get<1>(t).real - thrust::get<3>(t).real * thrust::get<0>(t).real - thrust::get<3>(t).imag * thrust::get<0>(t).imag;
	thrust::get<4>(t) = p1;
	thrust::get<5>(t) = p2;
}

polaritonSchrodinger::XYSysFunctor::XYSysFunctor(float CXLft): 
	CXLft_(CXLft) 
{
}

template<class Tuple>
 __host__ __device__
void polaritonSchrodinger::XYSysFunctor::operator()(Tuple t)
{
	Compl p1;
	Compl p2;

	p1.real = CXLft_ * thrust::get<0>(t).real + thrust::get<2>(t).imag/* + thrust::get<2>(t).real*/ + thrust::get<3>(t) * thrust::get<0>(t).imag;
	p1.imag = CXLft_ * thrust::get<0>(t).imag - thrust::get<2>(t).real/* + thrust::get<2>(t).imag*/ - thrust::get<3>(t) * thrust::get<0>(t).real;
	p2.real = CXLft_ * thrust::get<1>(t).real /*+ thrust::get<2>(t).imag - thrust::get<2>(t).real*/ + thrust::get<3>(t) * thrust::get<1>(t).imag;
	p2.imag = CXLft_ * thrust::get<1>(t).imag /*- thrust::get<2>(t).real - thrust::get<2>(t).imag*/ - thrust::get<3>(t) * thrust::get<1>(t).real;
	thrust::get<4>(t) = p1;
	thrust::get<5>(t) = p2;
}

void polaritonSchrodinger::createMat()
{
	std::vector<dtype> tex(Size_);
	std::vector<Compl> Sex(Size_);
	thrust::host_vector<Compl> pump(Size_);
	const int hxSize = xSize_ / 2;
	const int hySize = ySize_ / 2;
	#pragma omp parallel for
	for(int ikx = 0; ikx < xSize_; ++ikx) {
		for (int iky = 0; iky < ySize_; ++iky) {
			int hnkx = ikx <= hxSize ? 0 : xSize_;
			int hnky = iky <= hySize ? 0 : ySize_;
			float kx = (ikx - hnkx) * kxstep;
			float ky = (iky - hnky) * kystep;
			float ks = kx * kx + ky * ky;
			float k = std::sqrt(ks);
			int idx = ikx * ySize_ + iky;
			float CTex = Ex0 + h2mMx * ks; // + alpha * k;
			float km1 = 1.0f / k;
			if (ks > 1E-8) {
				Sex[idx].real = /*alpha * (kx * kx - ky * ky) * km1 * hbarm1;*/ beta * (kx * kx - ky * ky) * hbarm1;
				Sex[idx].imag = /*-2.0f * alpha * kx * ky * km1 * hbarm1;*/ -2.0 * beta * kx * ky * hbarm1;
			}
			else {
				Sex[idx].real = 0.0f;
				Sex[idx].imag = 0.0f;
			}
			tex[idx] = CTex * hbarm1;
			float Pol = Ex0 + h2mMx * ks;
			float Norm = sqrt2 * P0 * expf(-(k - kpa) * (k - kpa) * sigma * sigma * 0.25f);
			Norm *= Gamma * hbarm1 / (Pol * Pol + Gamma * Gamma);
			pump[idx].real = -Norm * Gamma;
			pump[idx].imag = Norm * Pol;
		}
	}
	tex_ = tex;
	Sex_ = Sex;
	pump_ = pump;
	
	cufftHandle plan;
	cufftPlan2d(&plan, xSize_, ySize_, CUFFT_C2C);
	cufftComplex* pp = reinterpret_cast<cufftComplex*>(const_cast<Compl*>(thrust::raw_pointer_cast(pump_.data())));
	cufftExecC2C(plan, pp, pp, CUFFT_INVERSE);
	thrust::transform(pump_.begin(), pump_.end(), pump_.begin(), MultFunctor(Sizem1_));

/*	pump = pump_;
	for (int ix = 0; ix < nkx; ++ix) {
		for (int iy = 0; iy < nky; ++iy) {
			int idx = ix * nky + iy;
			int hkx = ix <= nkx / 2 ? 0 : xSize_;
			int hky = iy <= nky / 2 ? 0 : ySize_;
			float x = (ix - hkx) * xstep;
			float y = (iy - hky) * ystep;
			std::printf("%6f\t%6f\t%6f\n", x, y, pump[idx].real * pump[idx].real + pump[idx].imag * pump[idx].imag);
		}
	}*/
	cufftDestroy(plan);
}

void polaritonSchrodinger::initUScat()
{
	for (int i = 0; i < Size_; ++i) {
		cr_[i] = dist_(rng_);
	}
	const Count hxSize = xSize_ / 2;
	const Count hySize = ySize_ / 2;
	#pragma omp parallel for 
	for(int ix = 0; ix < xSize_; ++ix) {
		for (int iy = 0; iy < ySize_; ++iy) {
			float result = 0.0f;
			int idx = ix * ySize_ + iy;
			int hkx = (ix <= hxSize) ? 0 : xSize_;
			int hky = (iy <= hySize) ? 0 : ySize_;
			float xc = (ix - hkx) * xstep;
			float yc = (iy - hky) * ystep;
			for(int ixp = 0; ixp < xSize_; ++ixp) {
				for (int iyp = 0; iyp < ySize_; ++iyp) {
					int idxp = ixp * ySize_ + iyp;
					float xcp = (ixp - hxSize) * xstep;
					float ycp = (iyp - hySize) * ystep;
					float dif = (xc - xcp) * (xc - xcp) + (yc - ycp) * (yc - ycp);
					result += cr_[idxp] * expf(-2.0f * dif / (lx * lx));
				}
			}
			UScatl_[idx] = 0.0; //2.0 * U0 * std::sqrt(xstep * ystep) * result / (hbar * lx * std::sqrt(Pi));
		}
	}
	/*for (int ix = 0; ix < nkx; ++ix) {
		for (int iy = 0; iy < nky; ++iy) {
			int idx = ix * nky + iy;
			int hkx = ix <= hxSize ? 0 : xSize_;
			int hky = iy <= hySize ? 0 : ySize_;
			float x = (ix - hkx) * xstep;
			float y = (iy - hky) * ystep;
			std::printf("%3f\t%3f\t%3f\n", x, y, thrust::abs(UScat[idx]));
		}
	}*/

	thrust::copy(UScatl_.begin(), UScatl_.end(), UScat_.begin());
	//UScat_ = UScatl_;
}

polaritonSchrodinger::MultFunctor::MultFunctor(dtype mult) : mult_(mult)
{
}

template<class Type>
__host__ __device__
Type polaritonSchrodinger::MultFunctor::operator()(const Type& x)
{
	Compl y = {mult_ * x.real, mult_ * x.imag};
	return y;
}

polaritonSchrodinger::SumMultFunctor::SumMultFunctor(dtype mult) : mult_(mult)
{
}

template<class Type>
__host__ __device__
Type polaritonSchrodinger::SumMultFunctor::operator()(const Type& x, const Type& y)
{
	return x + mult_ * y;
}

polaritonSchrodinger::~polaritonSchrodinger()
{
	delete[] n_;
	cufftDestroy(plan_);
}

observer::observer(Count xSize, Count ySize, shType& xInt, dtype tmax, dtype tdif) :
   	xSize_(xSize), ySize_(ySize), xInt_(xInt), xTemp_(2 * xSize_ * ySize_), /*kTemp_(4 * xSize_ * ySize_),*/ xDTemp_(2 * xSize_ * ySize_), tmax_ (tmax), tdif_(tdif), steps_(0), op_()
{
	/*Count n[2];
	n[0] = xSize_;
	n[1] = ySize_;
	Count Size = xSize_ * ySize_;
	int rank = 2;
	int batch = 2;
	cufftPlanMany(&plan_, rank, n, NULL, 1, Size, NULL, 1, Size, CUFFT_C2C, batch);*/
}

observer::~observer()
{
	//cufftDestroy(plan_);
}

void observer::initialize()
{
	thrust::fill(xInt_.begin(), xInt_.end(), 0.0f);
	steps_ = 0;
}


void observer::operator() (const sType& x, dtype t)
{
	const Count Size = nkx * nky;
//	thrust::copy(x.begin(), x.end(), xTemp_.begin());
/*	cufftComplex* xd = reinterpret_cast<cufftComplex*>(const_cast<dtype*>(thrust::raw_pointer_cast(x.data())));
	cufftComplex* kd = reinterpret_cast<cufftComplex*>(const_cast<dtype*>(thrust::raw_pointer_cast(kTemp_.data())));
	cufftExecC2C(plan_, xd, kd, CUFFT_FORWARD);*/

	cpType cphi1It(reinterpret_cast<const Compl*>(thrust::raw_pointer_cast(x.data())));
	cpType cphi2It = cphi1It + Size;
	cpType xend  = cphi2It + Size;

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(cphi1It, cphi2It, xDTemp_.begin(), xDTemp_.begin() + Size)), 
			thrust::make_zip_iterator(thrust::make_tuple(cphi2It, xend, xDTemp_.begin() + Size, xDTemp_.end())), OverFunctor(sqrt2));
/*	pType phi1It(reinterpret_cast<Compl*>(thrust::raw_pointer_cast(kTemp_.data())));
	pType phi2It = phi1It + Size;
	pType kend  = phi2It + Size;

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(phi1It, phi2It, xDTemp_.begin(), xDTemp_.begin() + Size)), 
			thrust::make_zip_iterator(thrust::make_tuple(phi2It, kend, xDTemp_.begin() + Size, xDTemp_.end())), OverFunctor(sqrt2));*/
	thrust::copy(xDTemp_.begin(), xDTemp_.end(), xTemp_.begin());
	thrust::transform(xInt_.begin(), xInt_.end(), xTemp_.begin(), xInt_.begin(), op_);
	steps_++;
}

void observer::normalize()
{
	thrust::constant_iterator<dtype> iter(1.0f / steps_);
	thrust::transform(xInt_.begin(), xInt_.end(), iter, xInt_.begin(), thrust::multiplies<dtype>());
}

observer::OverFunctor::OverFunctor(dtype sqrt2) : sqrt2_(sqrt2)
{
}
	
template<class Tuple>
 __host__ __device__
void observer::OverFunctor::operator()(Tuple t)
{
	Compl p1;
	Compl p2;

	p1.real = sqrt2_ * (thrust::get<0>(t).real + thrust::get<1>(t).real);
	p1.imag = sqrt2_ * (thrust::get<0>(t).imag + thrust::get<1>(t).imag);
	p2.real = sqrt2_ * (-thrust::get<0>(t).imag + thrust::get<1>(t).imag);
	p2.imag = sqrt2_ * (thrust::get<0>(t).real - thrust::get<1>(t).real);
	thrust::get<2>(t) = p1.real * p1.real + p1.imag * p1.imag;
	thrust::get<3>(t) = p2.real * p2.real + p2.imag * p2.imag;
}


