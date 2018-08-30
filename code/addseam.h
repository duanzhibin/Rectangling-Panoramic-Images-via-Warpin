#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int addSeam_wrap(Mat& img, Mat& mask, Mat xDispMap, Mat yDispMap,
		Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap,int axs);
