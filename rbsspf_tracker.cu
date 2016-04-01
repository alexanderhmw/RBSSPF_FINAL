#include"rbsspf_tracker.cuh"

using namespace std::chrono;

LaserScan h_scan;
double * d_scan;

EgoMotion h_egomotion;

int h_seed[MAXPN];
thrust::minstd_rand * d_rng=NULL;

TrackerParticle * h_tmpparticles;

//==============================================================================

extern "C" void cudaOpenTracker()
{
    srand(time(NULL));
    cudaCloseTracker();
    //==============================
    //initialize rand seed
    int * d_seed;
    cudaMalloc(&d_seed,sizeof(int)*MAXPN);
    thrust::generate(h_seed,h_seed+MAXPN,rand);
    cudaMemcpy(d_seed,h_seed,sizeof(int)*MAXPN,cudaMemcpyHostToDevice);
    cudaMalloc(&d_rng,sizeof(thrust::minstd_rand)*MAXPN);
    GetKernelDim_1D(blocks,threads,MAXPN);
    kernelSetupRandomSeed<<<blocks,threads>>>(d_seed,d_rng);
    CUDAFREE(d_seed);

    cudaMalloc(&d_scan,sizeof(double)*MAXBEAMNUM);

    h_tmpparticles=new TrackerParticle[MAXPN];
}

extern "C" void cudaCloseTracker()
{
    CUDAFREE(d_rng);
    CUDAFREE(d_scan);
    delete []h_tmpparticles;
}

//==============================================================================

extern "C" void cudaSetLaserScan(LaserScan & scan)
{
    h_scan=scan;
    cudaMemcpy(d_scan,h_scan.length,sizeof(double)*MAXBEAMNUM,cudaMemcpyHostToDevice);
    if(h_egomotion.validflag)
    {
        double tmpdx=h_egomotion.x-h_scan.x;
        double tmpdy=h_egomotion.y-h_scan.y;
        double c=cos(h_scan.theta);
        double s=sin(h_scan.theta);
        h_egomotion.dx=c*tmpdx+s*tmpdy;
        h_egomotion.dy=-s*tmpdx+c*tmpdy;
        h_egomotion.dtheta=h_egomotion.theta-h_scan.theta;
        h_egomotion.dt=h_scan.timestamp-h_egomotion.timestamp;
    }
    h_egomotion.x=h_scan.x;
    h_egomotion.y=h_scan.y;
    h_egomotion.theta=h_scan.theta;
    h_egomotion.timestamp=h_scan.timestamp;
    h_egomotion.validflag=1;
}

//==============================================================================

void SSPF_Motion(int & pnum, TrackerParticle * d_particles, int trackernum, TrackerSampleControl * h_controls, TrackerSampleControl * d_controls, TrackerParticle * d_tmpparticles, TrackerParticle * d_tmpparticles_forward,
                 int * h_beamcount, int * d_beamcount, double * h_weights, double * d_weights, int * h_controlids, int * d_controlids,
                 int * h_sampleids, int * d_sampleids, int * h_wcount, int * d_wcount, bool forwardflag)
{
    //update controls
    cudaMemcpy(d_controls,h_controls,sizeof(TrackerSampleControl)*trackernum,cudaMemcpyHostToDevice);

    //upsample
    int tmppnum=pnum*SPN;
    {
        GetKernelDim_1D(blocks,threads,tmppnum);
        kernelMotionUpSample<<<blocks,threads>>>(d_particles,d_controls,d_tmpparticles,d_tmpparticles_forward,tmppnum,d_rng,h_egomotion,h_scan.beamnum,d_beamcount);
    }

    //organize beam array
    int beamcount=hostCollectBeamCount(d_beamcount,h_beamcount,tmppnum);
    TrackerBeamEvaluator * d_beamevaluators;
    cudaMalloc(&d_beamevaluators,sizeof(TrackerBeamEvaluator)*beamcount);
    {
        GetKernelDim_1D(blocks,threads,tmppnum);
        kernelSetupBeamArray<<<blocks,threads>>>(d_beamcount,tmppnum,d_beamevaluators);
    }

    //measure
    {
        GetKernelDim_1D(blocks,threads,beamcount);
        kernelMeasureScan<<<blocks,threads>>>(d_beamevaluators,beamcount,d_tmpparticles_forward,d_controls,d_scan,h_scan.beamnum,1);
    }

    //accumulate weights
    {
        GetKernelDim_1D(blocks,threads,tmppnum);
        kernelAccumulateWeight<<<blocks,threads>>>(d_weights,d_controlids,d_tmpparticles,d_beamcount,tmppnum,d_beamevaluators);
    }

    cudaDeviceSynchronize();
    //cudaMemcpy(h_tmpparticles,d_tmpparticles,sizeof(TrackerParticle)*tmppnum,cudaMemcpyDeviceToHost);

    //downsample
    cudaMemcpy(h_weights,d_weights,sizeof(double)*tmppnum,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_controlids,d_controlids,sizeof(int)*tmppnum,cudaMemcpyDeviceToHost);
    pnum=0;
    int startid=0;
    while(startid<tmppnum)
    {
        int rqpn=hostDownSampleIDs(startid,h_controlids,h_weights,tmppnum,h_controls,h_sampleids+pnum,h_wcount+pnum,1);
        pnum+=rqpn;
    }
    cudaMemcpy(d_sampleids,h_sampleids,sizeof(int)*pnum,cudaMemcpyHostToDevice);
    cudaMemcpy(d_wcount,h_wcount,sizeof(int)*pnum,cudaMemcpyHostToDevice);
    {
        GetKernelDim_1D(blocks,threads,pnum);
        if(forwardflag)
        {
            kernelDownSample<<<blocks,threads>>>(d_particles,d_sampleids,d_wcount,pnum,d_tmpparticles_forward);
        }
        else
        {
            kernelDownSample<<<blocks,threads>>>(d_particles,d_sampleids,d_wcount,pnum,d_tmpparticles);
        }
    }

    CUDAFREE(d_beamevaluators);
}

void SSPF_Geometry(int & pnum, TrackerParticle * d_particles, int trackernum, TrackerSampleControl * h_controls, TrackerSampleControl * d_controls, TrackerParticle * d_tmpparticles,
                 int * h_beamcount, int * d_beamcount, double * h_weights, double * d_weights, int * h_controlids, int * d_controlids,
                 int * h_sampleids, int * d_sampleids, int * h_wcount, int * d_wcount)
{
    //update controls
    cudaMemcpy(d_controls,h_controls,sizeof(TrackerSampleControl)*trackernum,cudaMemcpyHostToDevice);

    //upsample
    int tmppnum=pnum*SPN;
    {
        GetKernelDim_1D(blocks,threads,tmppnum);
        kernelGeometryUpSample<<<blocks,threads>>>(d_particles,d_controls,d_tmpparticles,tmppnum,d_rng,h_scan.beamnum,d_beamcount);
    }

    //organize beam array
    int beamcount=hostCollectBeamCount(d_beamcount,h_beamcount,tmppnum);
    TrackerBeamEvaluator * d_beamevaluators;
    cudaMalloc(&d_beamevaluators,sizeof(TrackerBeamEvaluator)*beamcount);
    {
        GetKernelDim_1D(blocks,threads,tmppnum);
        kernelSetupBeamArray<<<blocks,threads>>>(d_beamcount,tmppnum,d_beamevaluators);
    }

    //measure
    {
        GetKernelDim_1D(blocks,threads,beamcount);
        kernelMeasureScan<<<blocks,threads>>>(d_beamevaluators,beamcount,d_tmpparticles,d_controls,d_scan,h_scan.beamnum,0);
    }

    //accumulate weights
    {
        GetKernelDim_1D(blocks,threads,tmppnum);
        kernelAccumulateWeight<<<blocks,threads>>>(d_weights,d_controlids,d_tmpparticles,d_beamcount,tmppnum,d_beamevaluators);
    }

    cudaDeviceSynchronize();

    //downsample
    cudaMemcpy(h_weights,d_weights,sizeof(double)*tmppnum,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_controlids,d_controlids,sizeof(int)*tmppnum,cudaMemcpyDeviceToHost);
    pnum=0;
    int startid=0;
    while(startid<tmppnum)
    {
        int rqpn=hostDownSampleIDs(startid,h_controlids,h_weights,tmppnum,h_controls,h_sampleids+pnum,h_wcount+pnum,0);
        pnum+=rqpn;
    }
    cudaMemcpy(d_sampleids,h_sampleids,sizeof(int)*pnum,cudaMemcpyHostToDevice);
    cudaMemcpy(d_wcount,h_wcount,sizeof(int)*pnum,cudaMemcpyHostToDevice);
    {
        GetKernelDim_1D(blocks,threads,pnum);
        kernelDownSample<<<blocks,threads>>>(d_particles,d_sampleids,d_wcount,pnum,d_tmpparticles);
    }

    CUDAFREE(d_beamevaluators);
}

extern "C" void cudaUpdateTracker(int trackernum, Tracker * trackers)
{
    if(trackernum<=0) return;
    //==============================
    //initialize sample control
    TrackerSampleControl * h_controls=new TrackerSampleControl[trackernum];
    TrackerSampleControl * d_controls=NULL;
    cudaMalloc(&d_controls,sizeof(TrackerSampleControl)*trackernum);

    //==============================
    //allocate memory of weight and sampleids
    int * h_sampleids=new int[RQPN*trackernum];
    int * d_sampleids=NULL;
    cudaMalloc(&d_sampleids,sizeof(int)*RQPN*trackernum);

    int * h_wcount=new int[RQPN*trackernum];
    int * d_wcount=NULL;
    cudaMalloc(&d_wcount,sizeof(int)*RQPN*trackernum);

    //==============================

    double * h_weights=new double[MAXPN*trackernum];
    double * d_weights=NULL;
    cudaMalloc(&d_weights,sizeof(double)*MAXPN*trackernum);

    int * h_beamcount=new int[MAXPN*trackernum];
    int * d_beamcount;
    cudaMalloc(&d_beamcount,sizeof(int)*MAXPN*trackernum);

    int * h_controlids=new int[MAXPN*trackernum];
    int * d_controlids;
    cudaMalloc(&d_controlids,sizeof(int)*MAXPN*trackernum);

    //==============================
    //allocate memory of paritlces and tmpparticles
    TrackerParticle * h_particles=new TrackerParticle[RQPN*trackernum];
    TrackerParticle * d_particles;
    cudaMalloc(&d_particles,sizeof(TrackerParticle)*RQPN*trackernum);
    TrackerParticle * d_tmpparticles;
    cudaMalloc(&d_tmpparticles,sizeof(TrackerParticle)*MAXPN*trackernum);
    TrackerParticle * d_tmpparticles_forward;
    cudaMalloc(&d_tmpparticles_forward,sizeof(TrackerParticle)*MAXPN*trackernum);

    //==============================
    int pnum;

    //==============================
    //Motion estimate

    //init
    double maxmotioniteration=hostInitMotionEstimation(trackers,trackernum,h_controls,h_particles,pnum);

    if(maxmotioniteration>=0)
    {
        cudaMemcpy(d_particles,h_particles,sizeof(TrackerParticle)*pnum,cudaMemcpyHostToDevice);

        //SSPF-loop
        for(int i=1;i<=maxmotioniteration;i++)
        {
            //SSPF_Motion
            SSPF_Motion(pnum,d_particles,trackernum,h_controls,d_controls,d_tmpparticles,d_tmpparticles_forward,h_beamcount,d_beamcount,h_weights,d_weights,h_controlids,d_controlids,h_sampleids,d_sampleids,h_wcount,d_wcount,0);

            //update controls
            for(int j=0;j<trackernum;j++)
            {
                if(h_controls[j].motioniteration>=1)
                {
                    h_controls[j].motioniteration--;
                    h_controls[j].motionoffset.a*=h_controls[j].motionzoom.a;
                    h_controls[j].motionoffset.v*=h_controls[j].motionzoom.v;
                    h_controls[j].motionoffset.omega*=h_controls[j].motionzoom.omega;
                    h_controls[j].motionanneal*=h_controls[j].motionannealratio;
                }
            }
        }

        //Final estimate
        //setup controls
        for(int j=0;j<trackernum;j++)
        {
            if(h_controls[j].motioniteration>=0)
            {
                h_controls[j].motioniteration=1;
                h_controls[j].motionoffset=MOTIONPREC;
                h_controls[j].motionanneal=1;
            }
        }

        //SSPF_Motion
        SSPF_Motion(pnum,d_particles,trackernum,h_controls,d_controls,d_tmpparticles,d_tmpparticles_forward,h_beamcount,d_beamcount,h_weights,d_weights,h_controlids,d_controlids,h_sampleids,d_sampleids,h_wcount,d_wcount,1);

        //Motion results
        cudaMemcpy(h_particles,d_particles,sizeof(TrackerParticle)*pnum,cudaMemcpyDeviceToHost);
        hostEstimateMotionTracker(h_particles,pnum,trackers);
    }

    //==============================
    //Geometry estimate

    //init
    double maxgeometryiteration=hostInitGeometryEstimation(trackers,trackernum,h_controls,h_particles,pnum);

    if(maxgeometryiteration>=0)
    {
        cudaMemcpy(d_particles,h_particles,sizeof(TrackerParticle)*pnum,cudaMemcpyHostToDevice);

        //SSPF-loop
        for(int i=1;i<=maxgeometryiteration;i++)
        {
            //SSPF_Geometry
            SSPF_Geometry(pnum,d_particles,trackernum,h_controls,d_controls,d_tmpparticles,h_beamcount,d_beamcount,h_weights,d_weights,h_controlids,d_controlids,h_sampleids,d_sampleids,h_wcount,d_wcount);

            //update controls
            for(int j=0;j<trackernum;j++)
            {
                if(h_controls[j].geometryiteration>=1)
                {
                    h_controls[j].geometryiteration--;
                    h_controls[j].geometryoffset.theta*=h_controls[j].geometryzoom.theta;
                    h_controls[j].geometryoffset.wl*=h_controls[j].geometryzoom.wl;
                    h_controls[j].geometryoffset.wr*=h_controls[j].geometryzoom.wr;
                    h_controls[j].geometryoffset.lf*=h_controls[j].geometryzoom.lf;
                    h_controls[j].geometryoffset.lb*=h_controls[j].geometryzoom.lb;
                    h_controls[j].geometryanneal*=h_controls[j].geometryannealratio;
                }
            }
        }

        //Final estimate
        //setup controls
        for(int j=0;j<trackernum;j++)
        {
            if(h_controls[j].geometryiteration>=0)
            {
                h_controls[j].geometryiteration=0;
                h_controls[j].geometryoffset=GEOMETRYPREC;
                h_controls[j].geometryanneal=1;
            }
        }

        //SSPF_Geometry
        SSPF_Geometry(pnum,d_particles,trackernum,h_controls,d_controls,d_tmpparticles,h_beamcount,d_beamcount,h_weights,d_weights,h_controlids,d_controlids,h_sampleids,d_sampleids,h_wcount,d_wcount);

        //Geometry results
        cudaMemcpy(h_particles,d_particles,sizeof(TrackerParticle)*pnum,cudaMemcpyDeviceToHost);
        hostEstimateGeometryTracker(h_particles,pnum,trackers,h_controls);
    }

    //==============================
    delete []h_controls;
    CUDAFREE(d_controls);

    delete []h_sampleids;
    CUDAFREE(d_sampleids);

    delete []h_wcount;
    CUDAFREE(d_wcount);

    delete []h_weights;
    CUDAFREE(d_weights);

    delete []h_beamcount;
    CUDAFREE(d_beamcount);

    delete []h_controlids;
    CUDAFREE(d_controlids);

    delete []h_particles;
    CUDAFREE(d_particles);
    CUDAFREE(d_tmpparticles);
    CUDAFREE(d_tmpparticles_forward);
}
