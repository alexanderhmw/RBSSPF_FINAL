#ifndef RBSSPF_MOTION_CUH
#define RBSSPF_MOTION_CUH

#include"rbsspf_share.cuh"

//====================================================

//1: init control and particles
__host__
double hostInitMotionEstimation(Tracker * trackers, int trackernum, TrackerSampleControl * controls, TrackerParticle * particles, int & pnum);

//2: upsample
__global__
void kernelMotionUpSample(TrackerParticle * particles, TrackerSampleControl * controls, TrackerParticle * tmpparticles, TrackerParticle * tmpparticles_forward, int tmppnum, thrust::minstd_rand * rng, EgoMotion egomotion, int beamnum, int * beamcount);

//8: estimate tracker
__host__
void hostEstimateMotionTracker(TrackerParticle * particles, int pnum, Tracker * trackers);

//====================================================

#endif // RBSSPF_MOTION_CUH
