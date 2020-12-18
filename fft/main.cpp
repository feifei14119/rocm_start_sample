#include "../common/utils.h"
#include "rocfft_kernel_16.h"

#include <fftw3.h>
#include "rocfft.h"

using namespace std;

//#define ASM_KERNEL
//#define HIP_KERNEL
//#define OBJ_V2
//#define OBJ_V3
//#define CMP_LLVM
//#define CMP_HCC

#define GROUP_NUM			(1)
#define WAVE_NUM			(1) // SIMD_PER_CU
#define ITERATION_TIMES		(1000)

// ==========================================================================================
#define FFT_LEN				(16)
#define FFT_THREAD_TILE 	(4)

float2 *h_x, *h_w, *h_y;				// cpu memory handle
float2 *d_x, *d_w, *d_y;				// gpu memory handle

// ==========================================================================================
#define MAX_WORK_GROUP_SIZE 1024

/* radix table: tell the FFT algorithms for size <= 4096 ; required by twiddle, passes, and kernel*/
struct SpecRecord
{
    size_t length;
    size_t workGroupSize;
    size_t numTransforms;
    size_t numPasses;
    size_t radices[12]; // Setting upper limit of number of passes to 12
};
inline const std::vector<SpecRecord>& GetRecord()
{
    static const std::vector<SpecRecord> specRecord = 
	{
        //  Length, WorkGroupSize (thread block size), NumTransforms , NumPasses,
        //  Radices
        //  vector<size_t> radices; NUmPasses = radices.size();
        //  Tuned for single precision on OpenCL stack; double precsion use the
        //  same table as single
        {4096, 256,  1, 3, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // pow2
        {2048, 256,  1, 4,  8,  8,  8, 4, 0, 0, 0, 0, 0, 0, 0, 0},
        {1024, 128,  1, 4,  8,  8,  4, 4, 0, 0, 0, 0, 0, 0, 0, 0},
        { 512,  64,  1, 3,  8,  8,  8, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 256,  64,  1, 4,  4,  4,  4, 4, 0, 0, 0, 0, 0, 0, 0, 0},
        { 128,  64,  4, 3,  8,  4,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {  64,  64,  4, 3,  4,  4,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {  32,  64, 16, 2,  8,  4,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {  16,  64, 16, 2,  4,  4,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {   8,  64, 32, 2,  4,  2,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {   4,  64, 32, 2,  2,  2,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {   2,  64, 64, 1,  2,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    return specRecord;
}
inline void DetermineSizes(const size_t& length, size_t& workGroupSize, size_t& numTrans)
{
    assert(MAX_WORK_GROUP_SIZE >= 64);

    if(length == 1) // special case
    {
        workGroupSize = 64;
        numTrans      = 64;
        return;
    }

    size_t baseRadix[]   = {13, 11, 7, 5, 3, 2}; // list only supported primes
    size_t baseRadixSize = sizeof(baseRadix) / sizeof(baseRadix[0]);

    size_t                   l = length;
    std::map<size_t, size_t> primeFactorsExpanded;
    for(size_t r = 0; r < baseRadixSize; r++)
    {
        size_t rad = baseRadix[r];
        size_t e   = 1;
        while(!(l % rad))
        {
            l /= rad;
            e *= rad;
        }

        primeFactorsExpanded[rad] = e;
    }

    assert(l == 1); // Makes sure the number is composed of only supported primes

    if(primeFactorsExpanded[2] == length) // Length is pure power of 2
    {
        if(length >= 1024)
        {
            workGroupSize = (MAX_WORK_GROUP_SIZE >= 256) ? 256 : MAX_WORK_GROUP_SIZE;
            numTrans      = 1;
        }
        else if(length == 512)
        {
            workGroupSize = 64;
            numTrans      = 1;
        }
        else if(length >= 16)
        {
            workGroupSize = 64;
            numTrans      = 256 / length;
        }
        else
        {
            workGroupSize = 64;
            numTrans      = 128 / length;
        }
    }
    else if(primeFactorsExpanded[3] == length) // Length is pure power of 3
    {
        workGroupSize = (MAX_WORK_GROUP_SIZE >= 256) ? 243 : 27;
        numTrans      = length >= 3 * workGroupSize ? 1 : (3 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[5] == length) // Length is pure power of 5
    {
        workGroupSize = (MAX_WORK_GROUP_SIZE >= 128) ? 125 : 25;
        numTrans      = length >= 5 * workGroupSize ? 1 : (5 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[7] == length) // Length is pure power of 7
    {
        workGroupSize = 49;
        numTrans      = length >= 7 * workGroupSize ? 1 : (7 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[11] == length) // Length is pure power of 11
    {
        workGroupSize = 121;
        numTrans      = length >= 11 * workGroupSize ? 1 : (11 * workGroupSize) / length;
    }
    else if(primeFactorsExpanded[13] == length) // Length is pure power of 13
    {
        workGroupSize = 169;
        numTrans      = length >= 13 * workGroupSize ? 1 : (13 * workGroupSize) / length;
    }
    else
    {
        size_t leastNumPerWI    = 1; // least number of elements in one work item
        size_t maxWorkGroupSize = MAX_WORK_GROUP_SIZE; // maximum work group size desired

        if(primeFactorsExpanded[2] * primeFactorsExpanded[3] == length)
        {
            if(length % 12 == 0)
            {
                leastNumPerWI    = 12;
                maxWorkGroupSize = 128;
            }
            else
            {
                leastNumPerWI    = 6;
                maxWorkGroupSize = 256;
            }
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[5] == length)
        {
            // NB:
            //   We found the below config leastNumPerWI 10 and maxWorkGroupSize
            //   128 works well for 1D cases 100 or 10000. But for single
            //   precision, 20/64 config is still better(>=) for most of the
            //   cases, especially for cases like 200, 800 with outplace large
            //   batch run.
            if((length % 20 == 0) && (length != 100))
            {
                leastNumPerWI    = 20;
                maxWorkGroupSize = 64;
            }
            else
            {
                leastNumPerWI    = 10;
                maxWorkGroupSize = 128;
            }
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 14;
            maxWorkGroupSize = 64;
        }
        else if(primeFactorsExpanded[3] * primeFactorsExpanded[5] == length)
        {
            leastNumPerWI    = 15;
            maxWorkGroupSize = 128;
        }
        else if(primeFactorsExpanded[3] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 21;
            maxWorkGroupSize = 128;
        }
        else if(primeFactorsExpanded[5] * primeFactorsExpanded[7] == length)
        {
            leastNumPerWI    = 35;
            maxWorkGroupSize = 64;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[3] * primeFactorsExpanded[5]
                == length)
        {
            leastNumPerWI    = 30;
            maxWorkGroupSize = 64;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[3] * primeFactorsExpanded[7]
                == length)
        {
            leastNumPerWI    = 42;
            maxWorkGroupSize = 60;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[5] * primeFactorsExpanded[7]
                == length)
        {
            leastNumPerWI    = 70;
            maxWorkGroupSize = 36;
        }
        else if(primeFactorsExpanded[3] * primeFactorsExpanded[5] * primeFactorsExpanded[7]
                == length)
        {
            leastNumPerWI    = 105;
            maxWorkGroupSize = 24;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[11] == length)
        {
            leastNumPerWI    = 22;
            maxWorkGroupSize = 128;
        }
        else if(primeFactorsExpanded[2] * primeFactorsExpanded[13] == length)
        {
            leastNumPerWI    = 26;
            maxWorkGroupSize = 128;
        }
        else
        {
            leastNumPerWI    = 210;
            maxWorkGroupSize = 12;
        }

        if(maxWorkGroupSize > MAX_WORK_GROUP_SIZE)
            maxWorkGroupSize = MAX_WORK_GROUP_SIZE;
        assert(leastNumPerWI > 0 && length % leastNumPerWI == 0);

        for(size_t lnpi = leastNumPerWI; lnpi <= length; lnpi += leastNumPerWI)
        {
            if(length % lnpi != 0)
                continue;

            if(length / lnpi <= MAX_WORK_GROUP_SIZE)
            {
                leastNumPerWI = lnpi;
                break;
            }
        }

        numTrans      = maxWorkGroupSize / (length / leastNumPerWI);
        numTrans      = numTrans < 1 ? 1 : numTrans;
        workGroupSize = numTrans * (length / leastNumPerWI);
    }

    assert(workGroupSize <= MAX_WORK_GROUP_SIZE);
}
std::vector<size_t> GetRadices(size_t length)
{
    std::vector<size_t> radices;

    // get number of items in this table
    std::vector<SpecRecord> specRecord  = GetRecord();
    size_t tableLength = specRecord.size();

    printf("tableLength=%zu\n", tableLength);
    for(int i = 0; i < tableLength; i++)
    {
        if(length == specRecord[i].length)
        { // if find the matched size

            size_t numPasses = specRecord[i].numPasses;
            printf("numPasses=%zu, table item %d \n", numPasses, i);
            for(int j = 0; j < numPasses; j++)
            {
                radices.push_back((specRecord[i].radices)[j]);
            }
            break;
        }
    }

    // if not in the table, then generate the radice order with the algorithm.
    if(radices.size() == 0)
    {
        size_t R = length;

        // Possible radices
        size_t cRad[]   = {13, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1}; // Must be in descending order
        size_t cRadSize = (sizeof(cRad) / sizeof(cRad[0]));

        size_t workGroupSize;
        size_t numTrans;
        // need to know workGroupSize and numTrans
        DetermineSizes(length, workGroupSize, numTrans);
        size_t cnPerWI = (numTrans * length) / workGroupSize;

        // Generate the radix and pass objects
        while(true)
        {
            size_t rad;

            // Picks the radices in descending order (biggest radix first) performanceCC
            // purpose
            for(size_t r = 0; r < cRadSize; r++)
            {

                rad = cRad[r];
                if((rad > cnPerWI) || (cnPerWI % rad))
                    continue;

                if(!(R % rad)) // if not a multiple of rad, then exit
                    break;
            }

            assert((cnPerWI % rad) == 0);

            R /= rad;
            radices.push_back(rad);

            assert(R >= 1);
            if(R == 1)
                break;

        } // end while
    } // end if(radices == empty)

	printf("radices size = %zu:\n",radices.size());
	for(std::vector<size_t>::const_iterator i = radices.begin(); i != radices.end(); i++)
		printf("\t%lu\n",*i);
	
    return radices;
}
class TwiddleTable
{
    size_t N;
    float2 * wc;

public:
    TwiddleTable(size_t length) : N(length)    {  wc = new float2[N]; }

    ~TwiddleTable()    { delete[] wc;  }

   
    float2* GenerateTwiddleTable(const std::vector<size_t>& radices)
    {
        const double TWO_PI = -6.283185307179586476925286766559;

        // Make sure the radices vector multiplication product up to N
        size_t sz = 1;
        for(std::vector<size_t>::const_iterator i = radices.begin(); i != radices.end(); i++)
        {
            sz *= (*i);
        }
        assert(sz == N);

        // Generate the table
        size_t L  = 1;
        size_t nt = 0;
        for(std::vector<size_t>::const_iterator i = radices.begin(); i != radices.end(); i++)
        {
            size_t radix = *i;

            L *= radix;

            // Twiddle factors
            for(size_t k = 0; k < (L / radix); k++)
            {
                double theta = TWO_PI * (k) / (L);

                for(size_t j = 1; j < radix; j++)
                {
                    double c = cos((j)*theta);
                    double s = sin((j)*theta);

                    // if (fabs(c) < 1.0E-12)    c = 0.0;
                    // if (fabs(s) < 1.0E-12)    s = 0.0;

                    wc[nt].x = c;
                    wc[nt].y = s;
                    nt++;
                }
            }
        } // end of for radices

        return wc;
    }
};

// ==========================================================================================
void InitHostMem()
{
	PrintStep1("Init Host Memory");

	h_x = (float2*)malloc(FFT_LEN * sizeof(float2));
	//h_w = (float2*)malloc(FFT_LEN * sizeof(float2));
	h_y = (float2*)malloc(FFT_LEN * sizeof(float2));

	for (unsigned int i = 0; i < FFT_LEN; i++)
	{
		h_x[i] = float2(i * 1.0f,  i * -0.1f);
		h_y[i] = float2(0, 0);
	}
	
	printf("twiddles_create_pr \n");
	std::vector<size_t> radices;
	radices = GetRadices(FFT_LEN);
	TwiddleTable twTable(FFT_LEN);

	h_w = twTable.GenerateTwiddleTable(radices);
	hipMalloc(&d_w, FFT_LEN * sizeof(float2));
	hipMemcpy(d_w, h_w, FFT_LEN * sizeof(float2), hipMemcpyHostToDevice);

	//printf("\nHost Vector A\n");	PrintHostData(h_A, FFT_LEN);
	//printf("\nHost Vector B\n");	PrintHostData(h_B, FFT_LEN);
}
void FreeHostMem()
{
	PrintStep1("Free Host Memory");

	free(h_x);
	free(h_w);
	free(h_y);
}

void InitDeviceMem()
{
	PrintStep1("Init Device Memory");

	PrintStep2("Malloc Device Memory");
	HIP_ASSERT(hipMalloc((void**)&d_x, FFT_LEN * sizeof(float2)));
	HIP_ASSERT(hipMalloc((void**)&d_w, FFT_LEN * sizeof(float2)));
	HIP_ASSERT(hipMalloc((void**)&d_y, FFT_LEN * sizeof(float2)));

	PrintStep2("Copy Host Memory To Device Memory");
	HIP_ASSERT(hipMemcpy(d_x, h_x, FFT_LEN * sizeof(float2), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_w, h_w, FFT_LEN * sizeof(float2), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(d_y, h_y, FFT_LEN * sizeof(float2), hipMemcpyHostToDevice));
}
void FreeDeviceMem()
{
	PrintStep1("Free Device Memory");

	hipFree(d_x);
	hipFree(d_w);
	hipFree(d_y);
}

// ==========================================================================================
void SetKernelArgs()
{
	PrintStep2("Setup Kernel Arguments");

	AddArg(d_x);
	AddArg(d_w);
	AddArg(d_y);
	AddArg(FFT_LEN);
}
void SetKernelWorkload()
{
	PrintStep2("Setup Kernel Workload");

	SetGroupSize(WAVE_SIZE * WAVE_NUM);
	SetGlobalSize(FFT_LEN);
}

// ==========================================================================================
void RunAsmCalculation()
{
	printf("\n---------------------------------------\n");
	PrintStep1("Run ASM Kernel");

	SetKernelArgs(); PrintKernelArgs();
	SetKernelWorkload(); PrintWorkload();
	LaunchKernel();
	FreeKernelArgs();

	printf("\nDevice Vector C\n"); PrintDeviceData(d_y, FFT_LEN);
}
void RunHipCalculation()
{
	printf("\n---------------------------------------\n");
	PrintStep1("Run HIP FFT");

	printf("twiddles_create_pr \n");
	std::vector<size_t> radices;
	radices = GetRadices(FFT_LEN);
	TwiddleTable twTable(FFT_LEN);

    // float tw
	float2 * twtc;
	float2 * dtw = NULL;
	twtc = twTable.GenerateTwiddleTable(radices);
	hipMalloc(&dtw, FFT_LEN * sizeof(float2));
	hipMemcpy(dtw, twtc, FFT_LEN * sizeof(float2), hipMemcpyHostToDevice);
	
    // debug
    const int dbg_len = 4096;
	float2 * hdebug = new float2[dbg_len];
	float2 * ddebug = NULL;
	hipMalloc(&ddebug, dbg_len * sizeof(float2));
    hipMemset(ddebug, 0, dbg_len * sizeof(float2));

	const size_t lengths[1] = {FFT_LEN};
	const size_t stride_in[2] = {1, FFT_LEN};
	const size_t stride_out[2] = {1, FFT_LEN};
	size_t * dlen = NULL;
	size_t * dstrin = NULL;
	size_t * dstrout = NULL;
	hipMalloc(&dlen, 1 * sizeof(size_t));
	hipMalloc(&dstrin, 2 * sizeof(size_t));
	hipMalloc(&dstrout, 2 * sizeof(size_t));
	hipMemcpy(dlen, lengths, 1 * sizeof(size_t), hipMemcpyHostToDevice);
	hipMemcpy(dstrin, stride_in, 2 * sizeof(size_t), hipMemcpyHostToDevice);
	hipMemcpy(dstrout, stride_out, 2 * sizeof(size_t), hipMemcpyHostToDevice);

	printf("gpu test\n");
	dim3 gp_sz = dim3(WAVE_SIZE * WAVE_NUM);
	dim3 glb_sz = dim3(FFT_LEN / FFT_THREAD_TILE);
	dim3 gp_num;
	gp_num.x = (glb_sz.x + gp_sz.x - 1) / gp_sz.x;
	gp_num.y = (glb_sz.y + gp_sz.y - 1) / gp_sz.y;
	gp_num.z = (glb_sz.z + gp_sz.z - 1) / gp_sz.z;
	printf("group  size = [%d, %d, %d]\n", gp_sz.x, gp_sz.y, gp_sz.z);
	printf("group  num  = [%d, %d, %d]\n", gp_num.x, gp_num.y, gp_num.z);
	printf("global size = [%d, %d, %d]\n", glb_sz.x, glb_sz.y, glb_sz.z);
	hipLaunchKernelGGL(fft_fwd_op_len16, gp_num, gp_sz, 0, 0, 
					dtw, FFT_LEN, 1, d_x, d_y, ddebug);

	printf("\nDevice debug\n"); PrintDeviceData(ddebug, FFT_LEN);
	printf("\nDevice Vector C\n"); PrintDeviceData(d_y, FFT_LEN);
}
void RunRocFFTCalculation()
{	
	printf("\n---------------------------------------\n");
	PrintStep1("Do rocFFT Calculation");

	rocfft_plan forward = NULL;
	rocfft_execution_info forwardinfo = NULL;
    size_t fbuffersize = 0;
    void* fbuffer = NULL;
	const size_t lengths[1] = {FFT_LEN};
	rocfft_plan_create(&forward,
					rocfft_placement_notinplace,
					rocfft_transform_type_complex_forward,
					rocfft_precision_single,
					1, // Dimension
					lengths,
					1, // Batch
					NULL);
	rocfft_execution_info_create(&forwardinfo);
	rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
	hipMalloc(&fbuffer, fbuffersize);
	rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
	rocfft_execute(forward, (void**)&d_x, (void**)&d_y, forwardinfo);

	printf("\nDevice Vector C\n"); PrintDeviceData(d_y, FFT_LEN);
}
void RunFFTWCalculation()
{
	printf("\n---------------------------------------\n");
	PrintStep1("Do FFTW Calculation");

    fftwf_complex *in, *out;
    fftwf_plan p;
    
    in  = (fftwf_complex*)h_x;
    out = (fftwf_complex*)h_y;

    p = fftwf_plan_dft_1d(FFT_LEN, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

	printf("\nHost Vector C\n");	PrintHostData(h_y, FFT_LEN);
    
    fftwf_destroy_plan(p);
}

// ==========================================================================================
void RunTest()
{
	printf("\n---------------------------------------\n");

	RunAsmCalculation();
	//RunHipCalculation();
	RunFFTWCalculation();
	RunRocFFTCalculation();

	printf("\n---------------------------------------\n");
}
int main(int argc, char *argv[])
{
	printf("\nHello ROCM.\n\n");
	InitHipRuntime();

	InitHostMem();
	InitDeviceMem();

	CreateAsmKernel("fft_len16", "fft_len16_v3.s");
	RunTest();

	FreeDeviceMem();
	FreeHostMem();

	ReleaseRuntime();
	printf("\nByeBye ROCM.\n\n");

	return 0;
}
