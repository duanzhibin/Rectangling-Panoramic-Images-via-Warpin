#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\core\utility.hpp>
#include<Eigen\Dense>

#include<math.h>

#include"LocalWarp.h"
#include"globalmesh.h"
#include"meshwarp.h"
#include"myFunctions.h"

using namespace cv;
using namespace std;
using namespace Eigen;


int mask_fg(Mat& rgbImg, int thrs, Mat &mask);
void my_imfillholes(Mat &src);

int main(int argc, char** argv){
	Mat image;                //原图
	image = imread("C:\\Users\\duan\\Desktop\\1_input.jpg", 1);

	if (image.empty()){
		cout << "图像加载失败" << endl;
		return -1;
	}
	Mat origImg = image / 255;

	//imshow("原图", image);
	//waitKey(10);
	//把原图压缩为1M；
	int rows = image.rows;
	int cols = image.cols;
	//	int color = image.channels;
	int megapixel = rows*cols;
	double scale = sqrt(double(800000) / megapixel);       //可改变压缩比
	Mat origImg1M;
	resize(image, origImg1M, Size(cols*scale, rows*scale), 0, 0, INTER_NEAREST);
	//imshow("缩小",origImg1M);
	//waitKey(10);
	int thrs = 253;
	Mat mask(Size(cols, rows), CV_8UC1);
	Mat mask1M(Size(cols*scale, rows*scale), CV_8UC1);                //mask
	mask_fg(image, thrs, mask);
	resize(mask, mask1M, Size(cols*scale, rows*scale), 0, 0, INTER_NEAREST);

#pragma omp parallel for
	for (int i = 0; i <mask1M.rows; i++){
		for (int j = 0; j < mask1M.cols; j++){
			if (mask1M.at<uchar>(i, j)*255>thrs)
				mask1M.at<uchar>(i, j) = 1;
			else
				mask1M.at<uchar>(i, j) = 0;

		}
	}
	
	//imshow("mask1M", mask1M * 255);                          //此处可以显示mask
	//waitKey(0);


	Mat OutImg(Size(cols*scale, rows*scale), CV_8UC3);
	Mat dispMap(Size(cols*scale, rows*scale), CV_32FC2);
	//LocalWarp::
	localWarping(origImg1M, mask1M, OutImg, dispMap);          //localWarping过程
	//cout << dispMap;
	//imshow("mask1M",mask1M*255);
	//waitKey(30);
	checkLocal(origImg1M,mask1M, dispMap);       
	//imshow("OutImg", OutImg);                                 //此处显示localwarping结束的图
	//waitKey(0);                                               //正确

	//imshow("origImg1M", origImg1M);
	//waitKey(30);


	Mat Vlocal, Vglobal;
	//globalWarp::
    globalmeshOpt(origImg1M, mask1M, dispMap,Vlocal,Vglobal);            //globalWarping过程
	
	Mat outputImg;
	//meshWarp::
    meshWarping(origImg1M, Vlocal, Vglobal, outputImg);

	imshow("outputImg", outputImg);
	waitKey(30);

	Vglobal.convertTo(Vglobal,CV_32S);
	Mat yVopt, xVopt;
	getchannel_int(Vglobal, 0, 2, yVopt);
	getchannel_int(Vglobal, 1, 2, xVopt);

	yVopt.convertTo(yVopt, CV_32F);
	xVopt.convertTo(xVopt, CV_32F);

	Mat gridmask, imageGridedx;
	drawGridmask(yVopt, xVopt, outputImg.rows,outputImg.cols, gridmask);
	drawGrid(gridmask, outputImg, imageGridedx);
	imshow("outputImageGridedx", imageGridedx);
	waitKey(0);

	double upScale = sqrt(double(megapixel) / 1000000);
	Mat finalImg;
	resize(outputImg, finalImg, Size(cols / upScale, rows / upScale), 0, 0, INTER_NEAREST);
	//imshow("最终结果", finalImg);
	//waitKey(0);
	
	return 0;
}


int mask_fg(Mat& rgbImg, int thrs, Mat &mask)
{
	Mat grayImg;
	cvtColor(rgbImg, grayImg, CV_BGR2GRAY);
	int rows = rgbImg.rows;
	int cols = rgbImg.cols;
	cout << rows << cols << endl;
	int i, j;
#pragma omp parallel for
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			if (grayImg.at<uchar>(i, j)>thrs-3)
				mask.at<uchar>(i, j) = 1;
			else
				mask.at<uchar>(i, j) = 0;
			
		}
	}


	my_imfillholes(mask);

	for (int i = 0; i < mask.rows; i++){
		mask.at<uchar>(i, 0) = 1;
		mask.at<uchar>(i, mask.cols - 1) = 1;
	}
	for (int i = 0; i < mask.cols; i++)
	{
		mask.at<uchar>(0, i) = 1;
		mask.at<uchar>(mask.rows - 1, i) = 1;
	}
	//imshow("mask", mask * 255);
	//waitKey(30);

	/*
	int g_nStructElementSize = 1; //结构元素(内核矩阵)的尺寸  
	//获取自定义核  
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
		Point(g_nStructElementSize, g_nStructElementSize));
	morphologyEx(mask, mask, MORPH_OPEN, element);
	
	
	 int g_nStructElementSize = 1; //结构元素(内核矩阵)的尺寸  
	//获取自定义核  
     Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
		Point(g_nStructElementSize, g_nStructElementSize));
	morphologyEx(mask, mask, MORPH_CLOSE, element);
	*/
	
	filter2D(mask, mask, mask.depth(), Mat::ones(7, 7, CV_8UC1));

	//mask = mask * 255;
	/*
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			if (mask.at<uchar>(i, j)>20)
				mask.at<uchar>(i, j) = 1;
			else
				mask.at<uchar>(i, j) = 0;

		}
	}*/
	filter2D(mask, mask, mask.depth(), Mat::ones(2, 2, CV_8UC1));
	for (i = 0; i < rows; i++){
#pragma omp parallel for
		for (j = 0; j < cols; j++){
			if (mask.at<uchar>(i, j)>1)
				mask.at<uchar>(i, j) = 1;
			else
				mask.at<uchar>(i, j) = 0;

		}
	}

	//imshow("mask", mask*255);
	//waitKey(30);
	
	//my_imfillholes(mask);
	//my_imfillholes(mask);
	return 0;
}

void my_imfillholes(Mat &src){
	// detect external contours
	//
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, cv::RETR_LIST, CHAIN_APPROX_NONE);
	//
	// fill external contours
	//
	if (!contours.empty() && !hierarchy.empty())
	{
		for (int idx =0; idx < contours.size(); idx++)
		{
			if (contours[idx].size() < 3000){
				drawContours(src, contours, idx, Scalar::all(0), CV_FILLED, 8);
			}
		}
	}
	//imshow("src", src*255);
	//waitKey(0);
}
