#ifndef RBSSPF_GEOMETRY_CUH
#define RBSSPF_GEOMETRY_CUH

#include"rbsspf_share.cuh"

//====================================================

//1: init control and particles
__host__
double hostInitGeometryEstimation(Tracker * trackers, int trackernum, TrackerSampleControl * controls, TrackerParticle * particles, int &pnum);

//2: upsample
__global__
void kernelGeometryUpSample(TrackerParticle * particles, TrackerSampleControl * controls, TrackerParticle * tmpparticles, int tmppnum, thrust::minstd_rand * rng, int beamnum, int * beamcount);

//8. estimate tracker
__host__
void hostEstimateGeometryTracker(TrackerParticle * particles, int pnum, Tracker * trackers, TrackerSampleControl * controls);

//====================================================

#endif // RBSSPF_GEOMETRY_CUH
