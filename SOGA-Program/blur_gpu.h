#ifndef __BLUR_GPU__
#define __BLUR_GPU__

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>


void inRange_gpu(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, Size size);


#endif
