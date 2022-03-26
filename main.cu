#include "operator.h"
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <cmath>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

int main(void)
{
	int Size = nkx * nky;
	sType inout(4 * Size);
	shType outCur(2 * Size);
	shType outTot(2 * Size);
	polaritonSchrodinger pSch(nkx, nky, 2, 2);
	observer obs(nkx, nky, outCur, tmax, tdif);

	thrust::fill(outTot.begin(), outTot.end(), 0.0f);
	thrust::plus<dtype> op;
	float tstart = 0.0f;

	boost::numeric::odeint::adams_bashforth_moulton<5 , sType, dtype, sType, dtype> abm;
	//boost::numeric::odeint::runge_kutta_dopri5< sType , dtype, sType , dtype > rkd;

	for (int count = 0; count < impl; ++count) {
		thrust::fill(inout.begin(), inout.end(), 0.0f);
		pSch.initUScat();
		abm.initialize(boost::ref(pSch), inout, tstart, tstep);
		size_t steps = boost::numeric::odeint::integrate_const(abm, boost::ref(pSch), inout, tstart, tmax - tdif, tstep);
	//size_t steps = boost::numeric::odeint::integrate_const(make_controlled( 1.0e-4, 1.0e-4, rkd), boost::ref(pSch), inout, tstart, tmax, tstep, boost::ref(obs));
		obs.initialize();
		float t = tmax - tdif;
		while(t < tmax) {
			steps = boost::numeric::odeint::integrate_const(abm, boost::ref(pSch), inout, t, t + tostep, tstep);
			obs(inout, t);
			t += tostep;
		}
		obs.normalize();
		//thrust::copy(inout.begin(), inout.end(), outCur.begin());
		thrust::transform(outTot.begin(), outTot.end(), outCur.begin(), outTot.begin(), op);
	}

	thrust::constant_iterator<dtype> iter(1.0f / impl);
	thrust::transform(outTot.begin(), outTot.end(), iter, outTot.begin(), thrust::multiplies<dtype>());

	for (int ix = 0; ix < nkx; ++ix) {
		for (int iy = 0; iy < nky; ++iy) {
			int idx = ix * nky + iy;
			int hkx = ix <= nkx / 2 ? 0 : nkx;
			int hky = iy <= nky / 2 ? 0 : nky;
			float x = (ix - hkx) * xstep;
			float y = (iy - hky) * ystep;
			//float p1r = sqrt2 * (outTot[2 * idx] + outTot[Size + 2 * idx]);
			//float p1i = sqrt2 * (outTot[2 * idx + 1] + outTot[Size + 2 * idx + 1]);
			//float p2r = sqrt2 * (-outTot[2 * idx + 1] + outTot[Size + 2 * idx + 1]);
			//float p2i = sqrt2 * (outTot[2 * idx] - outTot[Size + 2 * idx]);
			//float phi1Norm = outTot[2 * idx] * outTot[2 * idx] + outTot[2 * idx + 1] * outTot[2 * idx + 1]; 
			//float phi2Norm = outTot[Size + 2 * idx] * outTot[Size + 2 * idx] + outTot[Size + 2 * idx + 1] * outTot[Size + 2 * idx + 1];
			std::printf("%.6f\t%.6f\t%.6f\t%.6f\n", x, y, outTot[idx], outTot[Size + idx]); //p1r * p1r + p1i * p1i, p2r * p2r + p2i * p2i); //phi1Norm, phi2Norm);
		}
	}
}

