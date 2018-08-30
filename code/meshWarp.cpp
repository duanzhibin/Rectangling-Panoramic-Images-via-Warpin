#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <omp.h>

#include"meshwarp.h"
#include"myFunctions.h"

 

using namespace cv;
using namespace std;

int meshWarping(Mat& imgin, Mat& Vlocal, Mat& Vglobal, Mat& output_img){
	int rows = imgin.rows;
	int cols = imgin.cols;

	//int color = imgin.channels;
	Mat outputimg = Mat::zeros(rows, cols, CV_32SC3);
	Mat outputimgcount=Mat::zeros(rows, cols, CV_32SC3);

	

	Mat Vwarp = Vlocal.clone();
	Mat Vopt = Vglobal.clone() ;
	subtract(Vopt, 1, Vopt);

	//cout << Vlocal << Vopt << endl;

	int ygridNum = Vwarp.rows;
	int xgridNum = Vwarp.cols;

	Mat xVq(4, 1, CV_32SC1), yVq(4, 1, CV_32SC1);

	Mat xVopt, yVopt;
	getchannel_float(Vopt, 0, 2, yVopt);
	getchannel_float(Vopt, 1, 2, xVopt);

	Mat  xVo(4, 1, CV_32SC1), yVo(4, 1, CV_32SC1);
	Mat xVwarp, yVwarp;
	getchannel_float(Vwarp, 0, 2, yVwarp);
	getchannel_float(Vwarp, 1, 2, xVwarp);

	Mat Vq;                // (8, 1, CV_8UC1);
	Mat Vo;                // (8, 1, CV_8UC1);
	double xlength, ylength;
	double minVal, maxVal;
	double t2nstep, t1nstep;

	int quadrows = ygridNum - 1;
	int quadcols = xgridNum - 1;
	double v1w, v2w, v3w, v4w;
	Mat T(2, 8, CV_32FC1);


	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			//cout << i << "  " << j << endl;


			getvertices(i, j, yVopt, xVopt, yVq, xVq);                       //yVq,xVq (int)
			getvertices(i, j, yVwarp, xVwarp, yVo, xVo);

			//cout << yVq << xVq << endl;

			reshape(yVq, xVq, Vq);                                           //Vq (float)
			reshape(yVo, xVo, Vo);

			minMaxLoc(yVq, &minVal, &maxVal);
			ylength = double(maxVal - minVal);

			minMaxLoc(xVq, &minVal, &maxVal);
			xlength = double(maxVal - minVal);

			//cout << ylength << "  "<<xlength << endl;

			t2nstep = 1.0 / (xlength*4.0);
			t1nstep = 1.0 / (ylength*4.0);

			//cout << t2nstep << t1nstep << endl;


			for (double t1n = 0; t1n < 1;  t1n=t1n+t1nstep){
				for (double t2n = 0; t2n <1; t2n=t2n+t2nstep){
					//cout << t1n << endl;

					v1w = 1 - t1n - t2n + t1n*t2n;
					v2w = t2n - t1n*t2n;
					v3w = t1n - t1n*t2n;
					v4w = t1n*t2n;

					T.at<float>(0, 0) = v1w;  T.at<float>(1, 0) = 0;
					T.at<float>(0, 1) = 0;    T.at<float>(1, 1) = v1w;
					T.at<float>(0, 2) = v2w;  T.at<float>(1, 2) = 0;
					T.at<float>(0, 3) = 0;    T.at<float>(1, 3) = v2w;
					T.at<float>(0, 4) = v3w;  T.at<float>(1, 4) = 0;
					T.at<float>(0, 5) = 0;    T.at<float>(1, 5) = v3w;
					T.at<float>(0, 6) = v4w;  T.at<float>(1, 6) = 0;
					T.at<float>(0, 7) = 0;    T.at<float>(1, 7) = v4w;

					Mat pout = T*Vq;    // 2*8 8*1 = 2*1
					Mat pref = T*Vo;    // 2*8 8*1 = 2*1


					if (cvRound(pout.at<float>(1, 0)) >= 0 && cvRound(pout.at<float>(0, 0)) >= 0 && 
						cvRound(pref.at<float>(1, 0)) >= 0 && cvRound(pref.at<float>(0, 0)) >= 0){
#pragma omp parallel for
						for (int k = 0; k < 3; k++){
							outputimg.at<Vec3i>(cvRound(pout.at<float>(1, 0)), cvRound(pout.at<float>(0, 0)))[k]
								= outputimg.at<Vec3i>(cvRound(pout.at<float>(1, 0)), cvRound(pout.at<float>(0, 0)))[k]
								+ int(imgin.at<Vec3b>(cvRound(pref.at<float>(1, 0)), cvRound(pref.at<float>(0, 0)))[k]);

							outputimgcount.at<Vec3i>(cvRound(pout.at<float>(1, 0)), cvRound(pout.at<float>(0, 0)))[k]
								= outputimgcount.at<Vec3i>(cvRound(pout.at<float>(1, 0)), cvRound(pout.at<float>(0, 0)))[k] + 1;
						}
					}
					else{
						continue;
					}
				}
			}

		}
	}

	//cout << outputimgcount << endl;
	Mat outputimg_c = outputimg.clone();
	Mat outputimgcount_c = outputimgcount.clone();

	intToFloat(outputimg_c);
	intToFloat(outputimgcount_c);
	divide(outputimg_c, outputimgcount_c, output_img);
	output_img.convertTo(output_img, CV_8U);
	return 0;
}