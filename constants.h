#ifndef CONSTANTS_H_
#define CONSTANTS_H_
#include <cmath>

const float Pi = 3.14159f;
const float Ex0b = 0.0f; // Exciton energy at momentum zero (meV)
const float mx = 0.6f; // Exciton effective mass
const float tex = 5.3f; // Exciton lifetime in picoseconds
const float sigma = 0.4e3f; // Pump spot radius in nm
const float Epump = 0.0f; // Pump energy in meV from exciton bottom
const float kpx = 0.0f; // Pump momentum in x direction
const float kpy = 0.0f; // Pump momentum in y direction
const float kpa = std::sqrt(kpx * kpx + kpy * kpy);
const float Gamma = 0.2f; // Linewidth of the pump
const float E0 = Ex0b + Epump; // Zero of the energy
const float Ex0 = Ex0b - E0;
const float P0 = 1.0E+4f; //1E-2f; // Pump intensity
const float h2m = 1.05f * 1.05f * 1000.0f / (2.0f * 9.1f * 1.6f); //\hbar^2/2m in meV*nm^2
const float hbar = 1.05f / 1.6f; //\hbar in mev*ps
const float hbarm1 = 1.0f / hbar;
const float h2mMx = h2m / mx;
const float CXLft = -1.0f / (2.0f * tex);
const float alpha = 90.0f; //52.6f; // Excitonic SOC amplitude
const float beta = 10272.85f;
const float lx = 10.0f; //4.0f; // Standard deviation of disorder potential in nm
const float U0 = 0.05f; //5.0f; // Amplitude of disorder potential in meV
const float tstep = 1E-4f; // Time step
const float tostep = 1E-2f;
const float tmax = 40.0f;
const float tdif = 5.0f;
const int impl = 1;
const int nkx = 256;
const int nky = 256;

const float xrange = 12e3f;
const float yrange = 12e3f;
const float xstep = xrange / nkx;
const float ystep = yrange / nky;
const float kxstep = 2.0f * Pi / xrange;
const float kystep = 2.0f * Pi / yrange;
const int maxThrd = 32;

const float sqrt2 = 1.0f / std::sqrt(2);

#endif
