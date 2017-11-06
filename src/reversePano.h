#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include "opencv2/highgui/highgui.hpp"
#include "timer.h"
#include "device_launch_parameters.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

using namespace std;
using namespace cv;

namespace reversePano {
	int CPUmain(IplImage *inImageMat, IplImage **outImageMat);
	int GPUmain(IplImage *inImageMat, IplImage **outImageMat);
	IplImage* readImage(char *imageName);
	int writeImage(char *imageName, IplImage *image);
}