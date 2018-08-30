#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>

#include"LocalWarp.h"
#include"addseam.h"
#include"myFunctions.h"

using namespace cv;
using namespace std;

int findLB(Mat& mask, int& direction, int& p0, int& p1);
int findLS(int& p0, int& p1, int& len, Mat& v, int flag);

void getSub_ud(Mat& img, Mat& mask,Mat& xDispMap,Mat& yDispMap,Mat& subImg, Mat& subMask,Mat& subDispMap,Mat& subyDispMap,int p0, int p1, int anx);
void getSub_lr(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& subImg, Mat& subMask, Mat& subDispMap, Mat& subyDispMap, int p0, int p1, int anx);

void getUpdate_ud(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap, int p0, int p1);
void getUpdate_lr(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap, int p0, int p1);


int localWarping(Mat& origImg1M, Mat& mask1M, Mat& OutImg, Mat& out_dispMap){
	Mat img = origImg1M.clone();
	Mat mask = mask1M.clone();
	
	int cols = img.cols;
	int rows = img.rows;


	Mat xDispMap, yDispMap;
	meshgrid(cv::Range(1, cols), cv::Range(1, rows), xDispMap, yDispMap);
	
	//cout << xDispMap << yDispMap << endl;         //正确
	//cout << xDispMap.type()<<endl;

	Mat x, y;
	int d, p0, p1;
	Mat subImg, subMask;
	Mat subxDispMap, subyDispMap;
	Mat outImg, outMask;
	Mat outXDispMap, outYDispMap;
	int axs;

	while (1){
		findLB(mask, d, p0, p1);
		//cout << d <<" "<< p0<< " " << p1 << endl;
		if (d == 0){
			break;
		}
		switch (d)
		{
		case 1:{                  //up
			getSub_ud(img, mask, xDispMap, yDispMap, subImg, subMask, subxDispMap, subyDispMap, p0, p1, 1);
			//cout << subImg.rows << " " << subImg.cols << endl;
			meshgrid(cv::Range(1, cols), cv::Range(p0, p1), y, x);
			axs = 1;
			addSeam_wrap(subImg, subMask, subxDispMap, subyDispMap,
				outImg, outMask, outXDispMap, outYDispMap, axs);
			getUpdate_ud(img, mask, xDispMap, yDispMap, 
				outImg, outMask,outXDispMap,outYDispMap,p0,p1);
			break;

		}
		case 2:{                  //down		
			getSub_ud(img, mask, xDispMap, yDispMap, subImg, subMask, subxDispMap, subyDispMap, p0, p1, 2);
			meshgrid(cv::Range(1, cols), cv::Range(p0, p1), y, x);
			axs = 2;
		    addSeam_wrap(subImg, subMask, subxDispMap, subyDispMap,
				outImg, outMask, outXDispMap, outYDispMap, axs);
			getUpdate_ud(img, mask, xDispMap, yDispMap,
				outImg, outMask, outXDispMap, outYDispMap, p0, p1);
			break;

		}
		case 3:{                  //left
			getSub_lr(img, mask, xDispMap, yDispMap, subImg, subMask, subxDispMap, subyDispMap, p0, p1, 3);
			meshgrid(cv::Range(1, cols), cv::Range(p0, p1), x, y);
			axs = 3;
		    addSeam_wrap(subImg, subMask, subxDispMap, subyDispMap,
				outImg, outMask, outXDispMap, outYDispMap, axs);
			getUpdate_lr(img, mask, xDispMap, yDispMap,
				outImg, outMask, outXDispMap, outYDispMap, p0, p1);

			break;
		}
		case 4:{                  //right
			getSub_lr(img, mask, xDispMap, yDispMap, subImg, subMask, subxDispMap, subyDispMap, p0, p1, 4);
			meshgrid(cv::Range(1, cols), cv::Range(p0, p1), x, y);
			axs = 4;
			addSeam_wrap(subImg, subMask, subxDispMap, subyDispMap,
				outImg, outMask, outXDispMap, outYDispMap, axs);
			getUpdate_lr(img, mask, xDispMap, yDispMap,
				outImg, outMask, outXDispMap, outYDispMap, p0, p1);
			break;

		}
		default:
			break;
		}

		//subImg;                                     //get  subImg,
		//subMask;
		//subxDispMap;
		//subyDispMap;
		//img;                                       //updata
		//mask;
		//xDispMap;
		//yDispMap;
		//imshow("img", img);                           // 此处可以显示添加seam动态过程
		//waitKey(30);

	
		
	}
	//cout << yDispMap;
	//Try recovering image from displacement map

	//Output dispMap
	Mat dispMap;
	Mat xorigMap, yorigMap;
	meshgrid(cv::Range(1, cols), cv::Range(1, rows), xorigMap, yorigMap);
	//cout << xorigMap << yorigMap << endl; //正确

	

	Mat input1, input2;
	input2 = xDispMap - xorigMap;
	input1 = yDispMap - yorigMap;

	//cout << input2 <<"\n"<<input2 << endl;

	//cout << input1.type()<< endl;

	Mat out(input1.rows,input1.cols, CV_32FC2);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			out.at<Vec2f>(i, j)[0] = input1.at<int>(i, j);
			out.at<Vec2f>(i, j)[1] = input2.at<int>(i, j);
		}
	}
	//cout << out << endl;
	dispMap = out;
	//combin2Channel(input1,input2, dispMap);
	//for (int n = 0; n < rows; n++){
	//	for (int m = 0; m < cols; m++)
	//	{

	//		dispMap.at<Vec2d>[0] = xDispMap - xorigMap;
	//		dispMap.at<Vec2d>[1] = yDispMap - yorigMap;
	//	}
	//}

	//Output image
	OutImg = img.clone();
	out_dispMap = dispMap.clone();
	
	return 0;
}


int findLB(Mat& mask, int& direction, int& p0, int& p1){


	int rows = mask.rows;
	int cols = mask.cols;

	Mat mask_fb = mask.clone();
#pragma omp parallel for
	for (int i = 0; i < mask_fb.rows; i++){
		for (int j = 0; j <mask_fb.cols; j++){
	       if (mask_fb.at<uchar>(i, j) == 1){
				mask_fb.at<uchar>(i, j) = 1;
			}
			else{
				mask_fb.at<uchar>(i, j) = 0;
			}
		}
	}

	//imshow("mask_fb", mask_fb * 255);          // 此处可以显示添加seam动态过程
	//waitKey(30);

	Mat bnds_1;  //up
	Mat bnds_2;  //down
	Mat bnds_3;  //left
	Mat bnds_4;  //right

	bnds_1 = mask_fb.rowRange(0, 1).clone();
	bnds_2 = mask_fb.rowRange(rows - 1, rows).clone();
	bnds_3 = mask_fb.colRange(0, 1).clone();
	bnds_4 = mask_fb.colRange(cols - 1, cols).clone();
	//ranspose(bnds_3,bnds_3);
	//transpose(bnds_4,bnds_4);

	p0 = 0; p1 = 0; direction = 0;
	int longest = 0;
	int len, q0, q1;
	findLS( q0, q1,len, bnds_1, 1);         //up
	if (len > longest){
		longest = len;
		direction = 1;
		p0 = q0;
		p1 = q1;
	}
	
	findLS( q0, q1,len,bnds_2, 2);      //down
	if (len > longest){
		longest = len;
		direction = 2;
		p0 = q0;
		p1 = q1;
	}

	findLS( q0, q1,len,bnds_3, 3);     //left
	if (len > longest){
		longest = len;
		direction = 3;
		p0 = q0;
		p1 = q1;
	}

	findLS(q0, q1,len, bnds_4, 4);     //right
	if (len > longest){
		longest = len;
		direction = 4;
		p0 = q0;
		p1 = q1;
	}
	if (longest<2){
		direction = 0;
	}
	return 0;
}

int findLS(int& p0, int& p1, int& len, Mat& v, int flag){
	int i = 0;
	int rows, cols;
	int s1 = 0, s2 = 0;
	len = 0;
	p0 = 0, p1 = 0;

//	cout << v;
	switch (flag)
	{
	case 1:{
		rows = 1;
		cols = v.cols;
		int **x = new int*[rows];
		for (int i = 0; i < rows; i++){
			x[i] = new int[cols];
		}
		for (i = 0; i < cols + 1; i++){
			if (i == 0){
				x[0][i] = v.at<uchar>(0, i) - 0;
			}
			else if (i == cols)
			{
				x[0][i] = 0-int(v.at<uchar>(0, i-1));
			}
			else{
				x[0][i] =int(v.at<uchar>(0, i)) - int(v.at<uchar>(0, i - 1));
			}

			if (x[0][i] == 1){
				s1 = i;
			}
			else if ((x[0][i] == -1)){
				s2 = i;
			}

			if (s2 - s1>len){
				len = s2 - s1;
				p0 = s1;
				p1 = s2 - 1;
			}	
		}
		delete(x);
		break;
	}

	case 2:{
		rows = 1;
		cols = v.cols;
		int **x = new int*[rows];
		for (int i = 0; i < rows; i++){
			x[i] = new int[cols];
		}
		for (i = 0; i < cols + 1; i++){
			if (i == 0){
				x[0][i] = int(v.at<uchar>(0, i)) - 0;
			}
			else if (i == cols)
			{
				x[0][i] = 0 - int(v.at<uchar>(0, i-1));
			}
			else{
				x[0][i] = int(v.at<uchar>(0, i)) - int(v.at<uchar>(0, i - 1));
			}

			if (x[0][i] == 1){
				s1 = i;
			}
			else if ((x[0][i] == -1)){
				s2 = i;
			}
			if (s2 - s1>len){
				len = s2 - s1;
				p0 = s1;
				p1 = s2 - 1;
			}

		}

		delete(x);
		break;
	}

	case 3:{
		rows = v.rows;
		cols = 1;
		int **x = new int*[rows+1];
		for (int i = 0; i < rows+1; i++){
			x[i] = new int[cols];
		}

		for (i = 0; i < rows + 1; i++){
			if (i == 0){
				x[i][0] = int(v.at<uchar>(i, 0)) - 0;
			}
			else if (i == rows)
			{
				x[i][0] = 0 - int(v.at<uchar>(rows-1, 0));
			}
			else{
				x[i][0] = int(v.at<uchar>(i, 0)) - int(v.at<uchar>(i-1, 0));
			}

			if (x[i][0] == 1){
				s1 = i;
			}
			else if (x[i][0] == -1){
				s2 = i;
			}
			if (s2 - s1>len){
				len = s2 - s1;
				p0 = s1;
				p1 = s2 - 1;
			}
		}
		delete(x);
		//cout << len << endl;
		break;
	}

	case 4:{
		rows = v.rows;
		cols = 1;
		int **x = new int*[rows + 1];
		for (int i = 0; i < rows + 1; i++){
			x[i] = new int[cols];
		}

		for (i = 0; i < rows + 1; i++){
			if (i == 0){
				x[i][0] = int(v.at<uchar>(i, 0)) - 0;
			}
			else if (i == rows)
			{
				x[i][0] = 0 - int(v.at<uchar>(rows - 1, 0));
			}
			else{
				x[i][0] = int(v.at<uchar>(i, 0)) - int(v.at<uchar>(i - 1, 0));
			}

			if (x[i][0] == 1){
				s1 = i;
			}
			else if ((x[i][0] == -1)){
				s2 = i;
			}
			if (s2 - s1>len){
				len = s2 - s1;
				p0 = s1;
				p1 = s2 - 1;
			}
		}
		//cout << len << endl;
		delete(x);
		break;
	}
   
	default:
		break;
	}
	return 0;
}

//int addSeam_wrap(Mat& img, Mat& mask, Mat xDispMap, Mat yDispMap,
//	Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap){
//	return 0;
//}
/*
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}
*/

void getSub_ud(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& subImg, Mat& subMask, Mat& subxDispMap, Mat& subyDispMap, int p0, int p1, int anx){
	int cols = img.cols;
	int rows = img.rows;
	int i = 0, j = 0;

	Mat outimg(rows, p1 - p0 + 1, CV_8UC3);
	Mat outmask(rows, p1 - p0 + 1, CV_8UC1);
	Mat outxDispMap(rows, p1 - p0 + 1, CV_32FC1);
	Mat outyDispMap(rows, p1 - p0 + 1, CV_32FC1);
	for (i = 0; i < rows; i++){
		for (j = 0; j < p1 - p0 + 1; j++){
			outimg.at <Vec3b>(i, j) = img.at<Vec3b>(i, j + p0);
			outmask.at<uchar>(i, j) = mask.at<uchar>(i, j + p0);
			outxDispMap.at<float>(i, j) = xDispMap.at<int>(i, j + p0);
			outyDispMap.at<float>(i, j) = yDispMap.at<int>(i, j + p0);
		}
	}
	subImg = outimg;
	subMask = outmask;
	subxDispMap = outxDispMap;
	subyDispMap = outyDispMap;

}
void getSub_lr(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& subImg, Mat& subMask, Mat& subxDispMap, Mat& subyDispMap, int p0, int p1, int anx){
	int cols = img.cols;
	int rows = img.rows;
	int i = 0, j = 0;
	Mat outimg(p1 - p0 + 1, cols, CV_8UC3);
	Mat outmask(p1 - p0 + 1, cols, CV_8UC1);
	Mat outxDispMap(p1 - p0 + 1, cols, CV_32FC1);
	Mat outyDispMap(p1 - p0 + 1, cols, CV_32FC1);
	for (i = 0; i < p1 - p0 + 1; i++){
		for (j = 0; j < cols; j++){
			outimg.at <Vec3b>(i, j) = img.at<Vec3b>(i + p0, j);
			outmask.at<uchar>(i, j) = mask.at<uchar>(i + p0, j);
			outxDispMap.at<float>(i, j) = xDispMap.at<int>(i + p0, j);
			outyDispMap.at<float>(i, j) = yDispMap.at<int>(i + p0, j);
		}
	}
	subImg = outimg;
	subMask = outmask;
	subxDispMap = outxDispMap;
	subyDispMap = outyDispMap;

}
	
void getUpdate_ud(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap, int p0, int p1){
	int cols = img.cols;
	int rows = img.rows;
	int i = 0, j = 0;
	for (i = 0; i < rows; i++){
		for (j = 0; j < p1 - p0 + 1; j++){
			img.at<Vec3b>(i, j + p0) = outImg.at <Vec3b>(i, j);
			mask.at<uchar>(i, j + p0) = outMask.at<uchar>(i, j);
			xDispMap.at<int>(i, j + p0) = outXDispMap.at<float>(i, j);
			yDispMap.at<int>(i, j + p0) = outYDispMap.at<float>(i, j);
		}
	}
}

void getUpdate_lr(Mat& img, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap, int p0, int p1){
	int cols = img.cols;
	int rows = img.rows;
	int i = 0, j = 0;
	for (i = 0; i < p1 - p0 + 1; i++){
		for (j = 0; j < cols; j++){
			img.at<Vec3b>(i + p0, j) = outImg.at <Vec3b>(i, j);
			mask.at<uchar>(i + p0, j) = outMask.at<uchar>(i, j);
			xDispMap.at<int>(i + p0, j) = outXDispMap.at<float>(i, j);
			yDispMap.at<int>(i + p0, j) = outYDispMap.at<float>(i, j);
		}
	}

}

/*
void combin2Channel(Mat& inputImg1, Mat& inputImg2, Mat& outputImg){
	int rows = inputImg1.rows;
	int cols = inputImg1.cols;

	Mat out(rows, cols, CV_16UC2);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			out.at<Vec2b>(i, j)[0] = inputImg1.at<uchar>(i, j);
			out.at<Vec2b>(i, j)[1] = inputImg2.at<uchar>(i, j);
		}
	}

}
void combin3Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& outputImg){
	int rows = inputImg1.rows;
	int cols = inputImg1.cols;

	Mat out(rows, cols, CV_16UC3);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			out.at<Vec3b>(i, j)[0] = inputImg1.at<uchar>(i, j);
			out.at<Vec3b>(i, j)[1] = inputImg2.at<uchar>(i, j);
			out.at<Vec3b>(i, j)[2] = inputImg3.at<uchar>(i, j);
		}
	}

}
void combin4Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& inputImg4, Mat& outputImg){
	int rows = inputImg1.rows;
	int cols = inputImg1.cols;

	Mat out(rows, cols, CV_16UC4);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			out.at<Vec4b>(i, j)[0] = inputImg1.at<uchar>(i, j);
			out.at<Vec4b>(i, j)[1] = inputImg2.at<uchar>(i, j);
			out.at<Vec4b>(i, j)[2] = inputImg3.at<uchar>(i, j);
			out.at<Vec4b>(i, j)[3] = inputImg4.at<uchar>(i, j);
		}
	}
}
*/