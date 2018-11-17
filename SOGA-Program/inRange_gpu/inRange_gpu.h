#ifndef __INRANGE_GPU__
#define __INRANGE_GPU__

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>


void inRange_gpu(cv::cuda::GpuMat &src, cv::Scalar lowerb, cv::Scalar upperb,
                 cv::cuda::GpuMat &dst);


#endif
