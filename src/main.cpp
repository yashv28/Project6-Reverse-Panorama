#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include "reversePano.h"
#include "timer.h"

#define CPU 0
#define GPU 1
#define TIMER 1

using namespace cv;
using namespace std;


int main(int argc, char** argv) {
	int ret;
	cudaError_t cudaStatus;

	IplImage *inImageMat, *outImageMat;
	char* inImageName = "..\\images\\1024.jpg";
	char* outImageName = "..\\images\\o1024.jpg";

	inImageMat = reversePano::readImage(inImageName);
	if (inImageMat == NULL) {
		fprintf(stderr, "readImage failed!");
		return 1;
	}

	Mat I = cvarrToMat(inImageMat);
	resizeWindow("Input", 1366, 768);
	namedWindow("Input", WINDOW_NORMAL);
	imshow("Input", I);

#if CPU

#if TIMER
	CpuTimer timerCPU;
	timerCPU.Start();
#endif

	ret = reversePano::CPUmain(inImageMat, &outImageMat);
	if (ret != 0) {
		fprintf(stderr, "CPUmain failed!");
		return 1;
	}

#if TIMER
	timerCPU.Stop();
	printf("Total CPU time: %f s.\n", timerCPU.Elapsed());
#endif

#endif

#if GPU
#if TIMER
	CpuTimer timerGPU;
	timerGPU.Start();
#endif

	ret = reversePano::GPUmain(inImageMat, &outImageMat);
	if (ret != 0) {
		fprintf(stderr, "GPUmain failed!");
		return 1;
	}

#if TIMER
	timerGPU.Stop();
	printf("Total GPU time: %f s.\n", timerGPU.Elapsed());
#endif
#endif

	Mat m = cvarrToMat(outImageMat);
	resizeWindow("Output", 1366, 768);
	namedWindow("Output", WINDOW_NORMAL);
	imshow("Output", m);
	waitKey();
	ret = reversePano::writeImage(outImageName, outImageMat);
	if (ret != 0) {
		fprintf(stderr, "writeImage failed!");
		return 1;
	}

	return 0;
}