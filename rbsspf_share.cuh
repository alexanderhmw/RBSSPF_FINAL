#ifndef RBSSPF_CUH
#define RBSSPF_CUH

#include<cuda.h>
#include<cuda_runtime.h>

#include<thrust/random/linear_congruential_engine.h>
#include<thrust/random/uniform_real_distribution.h>
#include<thrust/random/normal_distribution.h>
#include<thrust/generate.h>

#include<random>
#include<time.h>

#define PI 3.14159265359
#define DEG2RAD(ang) (ang*PI/180)

#define MINSIGMA 1e-2
#define UNCERTAINTHRESH 0.4
#define MAXSIGMA 1e6

#define RQPN 64
#define SPN 4
#define MAXPN (SPN*RQPN)
#define MAXBEAMNUM 2048

#define CUDAFREE(pointer) if(pointer!=NULL){cudaFree(pointer);pointer=NULL;}

#define THREAD_1D 256
#define THREAD_2D 16
#define GetKernelDim_1D(numBlocks, threadsPerBlock, dim) int numBlocks=(dim+THREAD_1D-1)/THREAD_1D; int threadsPerBlock=THREAD_1D;
#define GetKernelDim_2D(numBlocks, threadsPerBlock, xdim, ydim) dim3 numBlocks(int((xdim+THREAD_2D-1)/THREAD_2D), int((ydim+THREAD_2D-1)/THREAD_2D)); dim3 threadsPerBlock(THREAD_2D, THREAD_2D);
#define GetThreadID_1D(id) int id=blockDim.x*blockIdx.x+threadIdx.x;
#define GetThreadID_2D(xid,yid) int xid=blockDim.x*blockIdx.x+threadIdx.x;int yid=blockDim.y*blockIdx.y+threadIdx.y;

#define NEARESTRING 3.35
#define MINBEAM 2
#define MAXBEAM 100

//==========================

#define SIGMA 0.01

#define COST0 1
#define WEIGHT0 -2
#define COST1 2
#define WEIGHT1 -8
#define COST2 0
#define WEIGHT2 0
#define COST3 1.6
#define WEIGHT3 -5.12

#define MARGIN0 0.2
#define MARGIN1 0.1
#define MARGIN2 0.1

//==========================

#define SSPFFLAG 1
#define REJECTFLAG 0

#define CALRATIO(ratio, vratio, maxratio, maxrange, minrange) \
    ratio=maxrange/minrange; vratio*=ratio; maxratio=ratio>maxratio?ratio:maxratio;
#define CALZOOM(zoom, maxrange, minrange, N) \
    zoom=log(maxrange/minrange)/log(2)/N;zoom=1/pow(2,zoom);

//==========================
#define MOTIONMIN {DEG2RAD(-60),-10,-0.5,DEG2RAD(-90)}
#define MOTIONMAX {DEG2RAD(60),30,0.5,DEG2RAD(90)}
#define MOTIONPREC {DEG2RAD(1),1,0.001,DEG2RAD(1)}
#define INITMOTIONOFFSET {DEG2RAD(60),20,0.5,DEG2RAD(90)}
#define UPDATEMOTIONOFFSET_SSPF {DEG2RAD(30),10,0.15,DEG2RAD(30)}
#define UPDATEMOTIONOFFSET_PF {DEG2RAD(10),3,0.05,DEG2RAD(10)}

#define GEOMETRYMIN {DEG2RAD(-30),0,0,0,0}
#define GEOMETRYMAX {DEG2RAD(30),3,3,5,5}
#define GEOMETRYPREC {DEG2RAD(1),0.1,0.1,0.1,0.1}
#define INITGEOMETRYOFFSET {DEG2RAD(30),1.5,1.5,2.5,2.5}
#define UPDATEGEOMETRYOFFSET {DEG2RAD(1),1.5,1.5,2.5,2.5}

struct GeometrySampleParam
{
    double theta;
    double wl,wr,lf,lb;
};

struct MotionSampleParam
{
    double a,v,k,omega;
};

struct TrackerSampleControl
{
    int id;
    bool pfflag;

    MotionSampleParam motionmin;
    MotionSampleParam motionmax;
    MotionSampleParam motionprec;
    MotionSampleParam motionoffset;
    MotionSampleParam motionzoom;

    double motioniteration;
    double motionanneal;
    double motionannealratio;

    GeometrySampleParam geometrymin;
    GeometrySampleParam geometrymax;
    GeometrySampleParam geometryprec;
    GeometrySampleParam geometryoffset;
    GeometrySampleParam geometryzoom;

    double geometryiteration;
    double geometryanneal;
    double geometryannealratio;

    int pnum;
};

//====================================================

enum TrackerStatus
{
    StatusInitGeometry,
    StatusInitMotion,
    StatusUpdateTracker_SSPF,
    StatusUpdateTracker_PF
};

struct TrackerState
{
    double x,y,theta;
    double wl,wr,lf,lb;
    double a,v,k,omega;
};

struct TrackerGeometry
{
    double cx[4],cy[4];//corners
    double dx[4],dy[4];//unit directions
    double cn[4],sa[4];//distance to origin and sin(alpha)
    int startid,startbeamid;
    int midid,midbeamid;
    int endid,endbeamid;
    int beamcount;
};

struct Tracker //CPU
{
    int id;
    TrackerStatus status;
    TrackerState mean;
    TrackerState sigma;
    double cx[4],cy[4];
};

struct TrackerParticle //GPU-core
{
    double weight;
    int count;
    TrackerState state;
    TrackerGeometry geometry;
    int controlid;
};

struct TrackerBeamEvaluator
{
    int tmppid;
    int beamdelta;
    double weight;
    bool validflag;
};

//====================================================

struct LaserScan //CPU
{
    double timestamp;
    double x,y,theta;
    int beamnum;
    double length[MAXBEAMNUM];
};

struct EgoMotion //GPU-cache
{
    bool validflag=0;
    double x,y,theta;
    double timestamp;
    double dx=0,dy=0,dtheta=0;
    double dt=0;
};

//====================================================

//0: init rng
__global__
void kernelSetupRandomSeed(int * seed, thrust::minstd_rand * rng); //0. MAXPN

//1: init control and particles (motion & geometry)
//int hostInitMotionEstimation(TrackerTracker * trackers, int trackernum, TrackerSampleControl * controls, TrackerParticle * particles);
//int hostInitGeometryEstimation(TrackerTracker * trackers, int trackernum, TrackerSampleControl * controls, TrackerParticle * particles);

//2: upsample (motion & geometry)
//void kernelMotionUpSample(TrackerParticle * particles, TrackerSampleControl * controls, TrackerParticle * tmpparticles, TrackerParticle * tmpparticles_forward, int tmppnum, thrust::minstd_rand * rng, EgoMotion egomotion, int beamnum, int * beamcount);
//void kernelGeometryUpSample(TrackerParticle * particles, TrackerSampleControl * controls, TrackerParticle * tmpparticles, int tmppnum, thrust::minstd_rand * rng, int beamnum, int * beamcount);

//3: collect beamcount and generate beamweight buffer
__host__
int hostCollectBeamCount(int * d_beamcount, int * h_beamcount, int tmppnum);

//4: setup beam array
__global__
void kernelSetupBeamArray(int * beamcount, int tmppnum, TrackerBeamEvaluator * beamevaluators); //3. tmppnum

//5: measure scan
__global__
void kernelMeasureScan(TrackerBeamEvaluator * beamevaluators, int beamcount, TrackerParticle * tmpparticles, TrackerSampleControl * controls, double * scan, int beamnum, bool motionflag); //4. beamcount

//6: accumulate beam weight
__global__
void kernelAccumulateWeight(double * weights, int * controlids, TrackerParticle * tmpparticles, int *beamcount, int tmppnum, TrackerBeamEvaluator * beamevaluators); //5. tmppnum

//7: get down sample ids
__host__
int hostDownSampleIDs(int & startid, int * controlids, double * weights, int tmppnum, TrackerSampleControl * controls, int * sampleids, int * wcount, bool motionflag); //6.

//8: down sample particles
__global__
void kernelDownSample(TrackerParticle * particles, int * sampleids, int *wcount, int pnum, TrackerParticle * tmpparticles); //7. pnum

//9: estimate tracker
//__host__
//void hostEstimateMotionTracker(TrackerParticle * particles, int pnum, TrackerTracker * tracker);
//__host__
//void hostEstimateGeometryTracker(TrackerParticle * particles, int pnum, TrackerTracker * tracker);

//====================================================

__host__ __device__
void deviceBuildModel(TrackerParticle & particle, int beamnum);

__host__
void hostBuildModel(Tracker & tracker);

//====================================================




#endif // RBSSPF_CUH
