#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#include"myFunctions.h"


#include<stdio.h>
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

void getvertices(int ygridID, int xgridID, Mat& yGrid, Mat& xGrid, Mat& yVq, Mat& xVq){
	//cout << ygridID << " " << xgridID << endl;
	yVq.at<int>(0, 0) = yGrid.at<float>(ygridID, xgridID);
	yVq.at<int>(1, 0) = yGrid.at<float>(ygridID, xgridID + 1);
	yVq.at<int>(2, 0) = yGrid.at<float>(ygridID + 1, xgridID);
	yVq.at<int>(3, 0) = yGrid.at<float>(ygridID + 1, xgridID + 1);

	xVq.at<int>(0, 0) = xGrid.at<float>(ygridID, xgridID);
	xVq.at<int>(1, 0) = xGrid.at<float>(ygridID, xgridID + 1);
	xVq.at<int>(2, 0) = xGrid.at<float>(ygridID + 1, xgridID);
	xVq.at<int>(3, 0) = xGrid.at<float>(ygridID + 1, xgridID + 1);
}


void reshape(Mat& yV, Mat& xV, Mat& V){
		Mat out(8, 1, CV_32FC1);
		out.at<float>(0, 0) = xV.at<int>(0, 0);
		out.at<float>(1, 0) = yV.at<int>(0, 0);
		out.at<float>(2, 0) = xV.at<int>(1, 0);
		out.at<float>(3, 0) = yV.at<int>(1, 0);
		out.at<float>(4, 0) = xV.at<int>(2, 0);
		out.at<float>(5, 0) = yV.at<int>(2, 0);
		out.at<float>(6, 0) = xV.at<int>(3, 0);
		out.at<float>(7, 0) = yV.at<int>(3, 0);
		V = out.clone();
}

void getchannel(Mat& input_img, int n_channel, int N, Mat& output_img){
	int rows = input_img.rows;
	int cols = input_img.cols;
	Mat out(rows, cols, CV_8UC1);
	if (N == 3){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<uchar>(i, j) = input_img.at<Vec3b>(i, j)[n_channel];
			}
		}
	}
	else if (N == 4){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<uchar>(i, j) = input_img.at<Vec4b>(i, j)[n_channel];
			}
		}
	}
	else if (N == 2){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<uchar>(i, j) = input_img.at<Vec2b>(i, j)[n_channel];
			}
		}
	}
	output_img = out.clone();
}
void getchannel_float(Mat& input_img, int n_channel, int N, Mat& output_img){
	int rows = input_img.rows;
	int cols = input_img.cols;
	Mat out(rows, cols, CV_32FC1);
	if (N == 3){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<float>(i, j) = input_img.at<Vec3f>(i, j)[n_channel];
			}
		}
	}
	else if (N == 4){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<float>(i, j) = input_img.at<Vec4f>(i, j)[n_channel];
			}
		}
	}
	else if (N == 2){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<float>(i, j) = input_img.at<Vec2f>(i, j)[n_channel];
			}
		}
	}
	output_img = out.clone();

}
void getchannel_int(Mat& input_img, int n_channel, int N, Mat& output_img){
	int rows = input_img.rows;
	int cols = input_img.cols;
	Mat out(rows, cols, CV_32SC1);
	if (N == 3){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<int>(i, j) = input_img.at<Vec3i>(i, j)[n_channel];
			}
		}
	}
	else if (N == 4){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<int>(i, j) = input_img.at<Vec4i>(i, j)[n_channel];
			}
		}
	}
	else if (N == 2){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<int>(i, j) = input_img.at<Vec2i>(i, j)[n_channel];
			}
		}
	}
	output_img = out.clone();

}



void combin2Channel(Mat& inputImg1, Mat& inputImg2, Mat& outputImg){
	int rows = inputImg1.rows;
	int cols = inputImg1.cols;

	if (inputImg1.type() == CV_8UC1){
		Mat out(rows, cols, CV_8UC2);
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<Vec2b>(i, j)[0] = inputImg1.at<uchar>(i, j);
				out.at<Vec2b>(i, j)[1] = inputImg2.at<uchar>(i, j);
			}
		}
		outputImg = out;
	}
	else if (inputImg1.type() == CV_32FC1){
		Mat out(rows, cols,CV_32FC2);
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<Vec2f>(i, j)[0] = inputImg1.at<float>(i, j);
				out.at<Vec2f>(i, j)[1] = inputImg2.at<float>(i, j);
			}
		}
		outputImg = out;
	}
	else if (inputImg1.type() == CV_32SC1){
		Mat out(rows, cols, CV_32SC2);
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				out.at<Vec2i>(i, j)[0] = inputImg1.at<int>(i, j);
				out.at<Vec2i>(i, j)[1] = inputImg2.at<int>(i, j);
			}
		}
		outputImg = out;
	}

}
void combin3Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& outputImg){
	int rows = inputImg1.rows;
	int cols = inputImg1.cols;

	Mat out(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			out.at<Vec3b>(i, j)[0] = inputImg1.at<uchar>(i, j);
			out.at<Vec3b>(i, j)[1] = inputImg2.at<uchar>(i, j);
			out.at<Vec3b>(i, j)[2] = inputImg3.at<uchar>(i, j);
		}
	}
	outputImg = out;
}
void combin4Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& inputImg4, Mat& outputImg){
	int rows = inputImg1.rows;
	int cols = inputImg1.cols;

	Mat out(rows, cols, CV_8UC4);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			out.at<Vec4b>(i, j)[0] = inputImg1.at<uchar>(i, j);
			out.at<Vec4b>(i, j)[1] = inputImg2.at<uchar>(i, j);
			out.at<Vec4b>(i, j)[2] = inputImg3.at<uchar>(i, j);
			out.at<Vec4b>(i, j)[3] = inputImg4.at<uchar>(i, j);
		}
	}
	outputImg = out;
}

void drawGridmask(Mat& ygrid,Mat& xgrid,int rows, int cols, Mat& gridmask)
{
	int xgridN = ygrid.cols;
	int ygridN = ygrid.rows;
    Mat outmask=Mat::zeros(rows, cols, CV_32FC1);

	double m;
    for (int y = 0; y < ygridN; y++){
		for (int x = 0; x < xgridN; x++){
			if (y != 0){
				if (ygrid.at<float>(y, x) != ygrid.at<float>(y - 1, x)){                                   //分母为0或者不为0 
					for (int i = ygrid.at<float>(y - 1, x); i <= ygrid.at<float>(y, x); i++){
						m = double(xgrid.at<float>(y, x) - xgrid.at<float>(y - 1, x)) /
							(ygrid.at<float>(y, x) - ygrid.at<float>(y - 1, x));
						outmask.at<float>(i, int(xgrid.at<float>(y - 1, x) +
							int(m*(i - ygrid.at<float>(y - 1, x))))) = 1;
					}
				}
				else{
					//for (int i = ygrid.at<float>(y - 1, x); i <= ygrid.at<float>(y, x); i++){
					//	outmask.at<float>(i, int(xgrid.at<float>(y - 1, x) +
					//		int(m*(i - ygrid.at<float>(y - 1, x))))) = 1;
					//}

				}
			}
			if (x != 0){
				if (xgrid.at<float>(y, x) != xgrid.at<float>(y, x - 1)){                     //分母为0或者不为0 
					for (int j = xgrid.at<float>(y, x - 1); j <= xgrid.at<float>(y, x); j++){
						m = double(ygrid.at<float>(y, x) - ygrid.at<float>(y, x - 1)) /
							(xgrid.at<float>(y, x) - xgrid.at<float>(y, x - 1));
						outmask.at<float>(int(ygrid.at<float>(y, x - 1) +
							int(m*(j - xgrid.at<float>(y, x - 1)))), j) = 1;
					}
				
				}
				else{
				//	for (int j = xgrid.at<float>(y, x - 1); j <= xgrid.at<float>(y, x); j++){
				//		outmask.at<float>(int(ygrid.at<float>(y, x - 1) +
				//			int(m*(j - xgrid.at<float>(y, x - 1)))), j) = 1;
				//	}
				}
			}
		}
	}
	//cout << outmask << endl;
	gridmask = outmask.clone();

}

void drawGrid(Mat& gridmask, Mat& image, Mat& outimage){
	Mat R, G, B;
	getchannel(image, 0, 3, R);
	getchannel(image, 1, 3, G);
	getchannel(image, 2, 3, B);
	for (int i = 0; i < gridmask.rows; i++){
		for (int j = 0; j < gridmask.cols; j++){
			if (gridmask.at<float>(i, j) == 1){
				R.at<uchar>(i, j) = 0;
				G.at<uchar>(i, j) = 255;
				B.at<uchar>(i, j) = 0;
			}
		}
	}

	combin3Channel(R, G, B, outimage);
}


void ucharToFloat(Mat& inputMat){
	if (inputMat.channels() == 1){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC1);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<float>(i, j) = inputMat.at<uchar>(i, j);
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 2){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC2);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec2f>(i, j)[0] = inputMat.at<Vec2b>(i, j)[0];
				out.at<Vec2f>(i, j)[1] = inputMat.at<Vec2b>(i, j)[1];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 3){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC3);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){

				out.at<Vec3f>(i, j)[0] = inputMat.at<Vec3b>(i, j)[0];
				out.at<Vec3f>(i, j)[1] = inputMat.at<Vec3b>(i, j)[1];
				out.at<Vec3f>(i, j)[2] = inputMat.at<Vec3b>(i, j)[2];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 4){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC4);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec3f>(i, j)[0] = inputMat.at<Vec4b>(i, j)[0];
				out.at<Vec3f>(i, j)[1] = inputMat.at<Vec4b>(i, j)[1];
				out.at<Vec3f>(i, j)[2] = inputMat.at<Vec4b>(i, j)[2];
				out.at<Vec3f>(i, j)[3] = inputMat.at<Vec4b>(i, j)[3];
			}
		}
		inputMat = out;
	}

}
void intToFloat(Mat& inputMat){
	if (inputMat.channels() == 1){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC1);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<float>(i, j) = inputMat.at<int>(i, j);
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 2){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC2);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec2f>(i, j)[0] = inputMat.at<Vec2i>(i, j)[0];
				out.at<Vec2f>(i, j)[1] = inputMat.at<Vec2i>(i, j)[1];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 3){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC3);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){

				out.at<Vec3f>(i, j)[0] = inputMat.at<Vec3i>(i, j)[0];
				out.at<Vec3f>(i, j)[1] = inputMat.at<Vec3i>(i, j)[1];
				out.at<Vec3f>(i, j)[2] = inputMat.at<Vec3i>(i, j)[2];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 4){
		Mat out(inputMat.rows, inputMat.cols, CV_32FC4);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec3f>(i, j)[0] = inputMat.at<Vec4i>(i, j)[0];
				out.at<Vec3f>(i, j)[1] = inputMat.at<Vec4i>(i, j)[1];
				out.at<Vec3f>(i, j)[2] = inputMat.at<Vec4i>(i, j)[2];
				out.at<Vec3f>(i, j)[3] = inputMat.at<Vec4i>(i, j)[3];
			}
		}
		inputMat = out;
	}

}

void floatTodouble(Mat& inputMat){
	if (inputMat.channels() == 1){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC1);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<double>(i, j) = inputMat.at<float>(i, j);
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 2){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC2);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec2d>(i, j)[0] = inputMat.at<Vec2f>(i, j)[0];
				out.at<Vec2d>(i, j)[1] = inputMat.at<Vec2f>(i, j)[1];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 3){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC3);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){

				out.at<Vec3d>(i, j)[0] = inputMat.at<Vec3f>(i, j)[0];
				out.at<Vec3d>(i, j)[1] = inputMat.at<Vec3f>(i, j)[1];
				out.at<Vec3d>(i, j)[2] = inputMat.at<Vec3f>(i, j)[2];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 4){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC4);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec3d>(i, j)[0] = inputMat.at<Vec4f>(i, j)[0];
				out.at<Vec3d>(i, j)[1] = inputMat.at<Vec4f>(i, j)[1];
				out.at<Vec3d>(i, j)[2] = inputMat.at<Vec4f>(i, j)[2];
				out.at<Vec3d>(i, j)[3] = inputMat.at<Vec4f>(i, j)[3];
			}
		}
		inputMat = out;
	}

}


void ucharTodouble(Mat& inputMat){
	if (inputMat.channels() == 1){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC1);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<double>(i, j) = inputMat.at<uchar>(i, j);
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 2){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC2);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec2d>(i, j)[0] = inputMat.at<Vec2b>(i, j)[0];
				out.at<Vec2d>(i, j)[1] = inputMat.at<Vec2b>(i, j)[1];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 3){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC3);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){

				out.at<Vec3d>(i, j)[0] = inputMat.at<Vec3b>(i, j)[0];
				out.at<Vec3d>(i, j)[1] = inputMat.at<Vec3b>(i, j)[1];
				out.at<Vec3d>(i, j)[2] = inputMat.at<Vec3b>(i, j)[2];
			}
		}
		inputMat = out;
	}
	else if (inputMat.channels() == 4){
		Mat out(inputMat.rows, inputMat.cols, CV_64FC4);
		for (int i = 0; i < inputMat.rows; i++){
			for (int j = 0; j < inputMat.cols; j++){
				out.at<Vec3d>(i, j)[0] = inputMat.at<Vec4b>(i, j)[0];
				out.at<Vec3d>(i, j)[1] = inputMat.at<Vec4b>(i, j)[1];
				out.at<Vec3d>(i, j)[2] = inputMat.at<Vec4b>(i, j)[2];
				out.at<Vec3d>(i, j)[3] = inputMat.at<Vec4b>(i, j)[3];
			}
		}
		inputMat = out;
	}

}


void blkdiag(Mat& input1, Mat &input2, Mat& output){
	//cout << input1.rows << input1.cols<<endl;
	if (input1.type() == CV_8UC1){
		Mat out = Mat::zeros(input1.rows+input2.rows, input1.cols+input2.cols, CV_8UC1);
		for (int i = 0; i < input1.rows; i++){
			for (int j = 0; j < input1.cols; j++){
				out.at<uchar>(i, j) = input1.at<uchar>(i, j);
			}
		}
		for (int i = 0; i < input2.rows; i++){
			for (int j = 0; j < input2.cols; j++){
				out.at<uchar>(i + input1.rows, j + input1.cols) = input2.at<uchar>(i, j);
			}
		}
		output = out;
	}
	else if (input1.type() == CV_32FC1){
		Mat out = Mat::zeros(input1.rows+input2.rows,input1.cols+ input2.cols, CV_32FC1);
		//cout << out.rows << out.cols << endl;
		for (int i = 0; i < input1.rows; i++){
			for (int j = 0; j < input1.cols; j++){
				out.at<float>(i, j) = input1.at<float>(i, j);
				//cout << i << j << endl;
			}
		}
		for (int i = 0; i < input2.rows; i++){
			for (int j = 0; j < input2.cols; j++){
				out.at<float>(i + input1.rows, j + input1.cols) = input2.at<float>(i, j);
			}
		}
		output = out;
	}
	else if (input1.type() == CV_32SC1){
		Mat out = Mat::zeros(input1.rows+input2.rows, input1.cols+input2.cols, CV_32SC1);
		for (int i = 0; i < input1.rows; i++){
			for (int j = 0; j < input1.cols; j++){
				out.at<int>(i, j) = input1.at<int>(i, j);
			}
		}
		for (int i = 0; i < input2.rows; i++){
			for (int j = 0; j < input2.cols; j++){
				out.at<int>(i + input1.rows, j + input1.cols) = input2.at<int>(i, j);
			}
		}
		output = out;
	}

}


void checkInterSection(Mat& lineA, Mat& lineB, int& isInterSect, Mat& p){

	//cout << lineA << "\n" << lineB << endl;   //           2018/8/8  出错


	intToFloat(lineA);
//	intToFloat(lineB);

	Mat A1 = lineA.rowRange(1, 2).clone();
	Mat A2 = lineA.rowRange(0, 1).clone();
	Mat deltaA = A1 - A2;
	//cout << deltaA;
	Mat B1 = lineB.rowRange(1, 2).clone();
	Mat B2 = lineB.rowRange(0, 1).clone();
	Mat deltaB = B1 - B2;
	//cout << deltaB;
	//cout << 1 << endl;
	Mat deltaA_t = deltaA.t();
	//cout << 2 << endl;
	Mat deltaB_t = deltaB.t();
	Mat M;
	hconcat(-deltaA_t, deltaB_t, M);

	//cout << M<<M.rows;
	Mat t;
	Mat M_inv = M.inv();
	Mat x = lineA.rowRange(0, 1) - lineB.rowRange(0, 1);
	//cout << x;
	x = x.t();
	t = M_inv*x;

	//cout << t << endl;
	int t1 = 0, t0 = 0;
	int rows = t.rows;
	int cols = t.cols;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if (t.at<float>(i, j) <= 1){
				t1++;
			}
			if (t.at<float>(i, j) >= 0){
				t0++;
			}
		}
	}
	isInterSect = false;
	if (t1 == 2 && t0 == 2){
		isInterSect = true;
	}

	p = (lineA.rowRange(0, 1) + t.at<float>(0, 0)*deltaA);


}


void getLinTrans(int pstart_y, int pstart_x, Mat& yVq, Mat& xVq, Mat& T){
	// V is a 8*1 vector, p is 2*1  T is 2*8
	//
	Mat V(8, 1, CV_32FC1);
	V.at<float>(0, 0) = xVq.at<int>(0, 0); V.at<float>(1, 0) = yVq.at<int>(0, 0);
	V.at<float>(2, 0) = xVq.at<int>(1, 0); V.at<float>(3, 0) = yVq.at<int>(1, 0);
	V.at<float>(4, 0) = xVq.at<int>(2, 0); V.at<float>(5, 0) = yVq.at<int>(2, 0);
	V.at<float>(6, 0) = xVq.at<int>(3, 0); V.at<float>(7, 0) = yVq.at<int>(3, 0);

	Mat v1(2, 1, CV_32FC1), v2(2, 1, CV_32FC1), v3(2, 1, CV_32FC1), v4(2, 1, CV_32FC1);
	v1.at<float>(0, 0) = xVq.at<int>(0, 0); v1.at<float>(1, 0) = yVq.at<int>(0, 0);
	v2.at<float>(0, 0) = xVq.at<int>(1, 0); v2.at<float>(1, 0) = yVq.at<int>(1, 0);
	v3.at<float>(0, 0) = xVq.at<int>(2, 0); v3.at<float>(1, 0) = yVq.at<int>(2, 0);
	v4.at<float>(0, 0) = xVq.at<int>(3, 0); v4.at<float>(1, 0) = yVq.at<int>(3, 0);

	Mat v21 = v2 - v1, v31 = v3 - v1, v41 = v4 - v1;

	Mat p(2, 1, CV_32FC1);
	p.at<float>(0, 0) = pstart_x;  p.at<float>(1, 0) = pstart_y;
	Mat p1 = p - v1;

	double a1 = v31.at<float>(0, 0), a2 = v21.at<float>(0, 0),          //x
		a3 = v41.at<float>(0, 0) - v31.at<float>(0, 0) - v21.at<float>(0, 0);
	double b1 = v31.at<float>(1, 0), b2 = v21.at<float>(1, 0),      //y
		b3 = v41.at<float>(1, 0) - v31.at<float>(1, 0) - v21.at<float>(1, 0);
	
	
	double px = p1.at<float>(0, 0), py = p1.at<float>(1, 0);

	Mat tvec, mat_t;
	double t1n, t2n;
	double a, b, c;
	if (a3 == 0 && b3 == 0){
		hconcat(v31, v21, mat_t);
		tvec = mat_t.inv()*p1;
		t1n = tvec.at<float>(0, 0);
		t2n = tvec.at<float>(1, 0);
	}
	else{
		a = (b2*a3 - a2*b3);
		b = (-a2*b1 + b2*a1 + px*b3 - a3*py);
		c = px*b1 - py*a1;
		if (a == 0){
			t2n = -c / b;
		}
		else{                                   //此处改动，出现(b*b - 4 * a*c) < 0 时，无法开方，改为开方结果为 0
			if ((b*b - 4 * a*c) > 0){
				t2n = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);
			}
			else{
				t2n = (-b - 0) / (2 * a);
			}
			//cout << t2n << endl;
		}

		if (abs(a1 + t2n*a3) <= 0.0000001){
			t1n = (py - t2n*b2) / (b1 + t2n*b3);
		}
		else{
			t1n = (px - t2n*a2) / (a1 + t2n*a3);
		}

	}

	//cout << t2n << endl;
	Mat m1 = v1 + t1n*(v3 - v1);
	Mat m4 = v2 + t1n*(v4 - v2);
	Mat ptest = m1 + t2n*(m4 - m1);

	double v1w = 1 - t1n - t2n + t1n*t2n;
	double v2w = t2n - t1n*t2n;
	double v3w = t1n - t1n*t2n;
	double v4w = t1n*t2n;

	Mat out(2, 8, CV_32FC1);
	out.at<float>(0, 0) = v1w;  out.at<float>(1, 0) = 0;
	out.at<float>(0, 1) = 0;    out.at<float>(1, 1) = v1w;
	out.at<float>(0, 2) = v2w;  out.at<float>(1, 2) = 0;
	out.at<float>(0, 3) = 0;    out.at<float>(1, 3) = v2w;
	out.at<float>(0, 4) = v3w;  out.at<float>(1, 4) = 0;
	out.at<float>(0, 5) = 0;    out.at<float>(1, 5) = v3w;
	out.at<float>(0, 6) = v4w;  out.at<float>(1, 6) = 0;
	out.at<float>(0, 7) = 0;    out.at<float>(1, 7) = v4w;
	T = out.clone();
	assert(norm(T*V - p) < 0.0001);

}


void checkLocal(Mat& image,Mat &mask, Mat& dispMap){
	Mat outMask=mask.clone();
	Mat outImg = image.clone();
	for (int i = 0; i < outMask.rows; i++){
		for (int j = 0; j < outMask.cols; j++){
			//cout << int(i + dispMap.at<Vec2f>(i, j)[0]) <<"  "<< int(j + dispMap.at<Vec2f>(i, j)[1]) << endl;
			outMask.at<uchar>(i, j) = mask.at<uchar>(i + dispMap.at<Vec2f>(i, j)[0], j + dispMap.at<Vec2f>(i, j)[1]);
			outImg.at<Vec3b>(i, j) = image.at<Vec3b>(i + dispMap.at<Vec2f>(i, j)[0], j + dispMap.at<Vec2f>(i, j)[1]);
		}
	}
	//imshow("checkDispMap", outMask*255);
	//waitKey(30);
	imshow("checkDispMap2",outImg);
	waitKey(30);
}


int checkIsIn(Mat& yVq, Mat& xVq, int pstartx, int pstarty, int pendx, int pendy){
	
	int min_x = min(pstartx, pendx);
	int min_y = min(pstarty, pendy);


	int max_x = max(pstartx, pendx);
	int max_y = max(pstarty, pendy);

	if ((min_x < xVq.at<int>(0, 0) && min_x < xVq.at<int>(2, 0)) || max_x>xVq.at<int>(1, 0) && max_x>xVq.at<int>(3, 0))
	{
		return 0;
	}
	else if ((min_y < yVq.at<int>(0, 0) && min_y < yVq.at<int>(1, 0))||( max_y>yVq.at<int>(2, 0) && max_y>yVq.at<int>(3, 0)))
	{
		return 0;
	}
	else
	{
		return 1;
	}
}