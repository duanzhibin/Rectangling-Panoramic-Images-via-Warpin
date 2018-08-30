#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

void getvertices(int ygridID, int xgridID, Mat& yGrid, Mat& xGrid, Mat& yVq, Mat& xVq){
	Mat vq(4, 1, CV_8UC1);
	yVq.at<uchar>(0, 0) = yGrid.at<uchar>(ygridID, xgridID);
	yVq.at<uchar>(1, 0) = yGrid.at<uchar>(ygridID, xgridID + 1);
	yVq.at<uchar>(2, 0) = yGrid.at<uchar>(ygridID + 1, xgridID);
	yVq.at<uchar>(3, 0) = yGrid.at<uchar>(ygridID + 1, xgridID + 1);

	xVq.at<uchar>(0, 0) = xGrid.at<uchar>(ygridID, xgridID);
	xVq.at<uchar>(1, 0) = xGrid.at<uchar>(ygridID, xgridID + 1);
	xVq.at<uchar>(2, 0) = xGrid.at<uchar>(ygridID + 1, xgridID);
	xVq.at<uchar>(3, 0) = xGrid.at<uchar>(ygridID + 1, xgridID + 1);
}



void getLinTrans(int pstart_y, int pstart_x, Mat& yVq, Mat& xVq, Mat& T){
	// V is a 8*1 vector, p is 2*1  T is 2*8
	//
	Mat V(8, 1, CV_8SC1);
	V.at<uchar>(0, 0) = yVq.at<uchar>(0, 0); V.at<uchar>(1, 0) = xVq.at<uchar>(0, 0);
	V.at<uchar>(2, 0) = yVq.at<uchar>(1, 0); V.at<uchar>(3, 0) = xVq.at<uchar>(1, 0);
	V.at<uchar>(4, 0) = yVq.at<uchar>(2, 0); V.at<uchar>(5, 0) = xVq.at<uchar>(2, 0);
	V.at<uchar>(6, 0) = yVq.at<uchar>(3, 0); V.at<uchar>(7, 0) = xVq.at<uchar>(3, 0);

	Mat v1(2, 1, CV_8UC1), v2(2, 1, CV_8UC1), v3(2, 1, CV_8UC1), v4(2, 1, CV_8UC1);
	v1.at<uchar>(0, 0) = yVq.at<uchar>(0, 0); v1.at<uchar>(1, 0) = xVq.at<uchar>(0, 0);
	v2.at<uchar>(0, 0) = yVq.at<uchar>(1, 0); v1.at<uchar>(1, 0) = xVq.at<uchar>(1, 0);
	v3.at<uchar>(0, 0) = yVq.at<uchar>(2, 0); v1.at<uchar>(1, 0) = xVq.at<uchar>(2, 0);
	v4.at<uchar>(0, 0) = yVq.at<uchar>(3, 0); v1.at<uchar>(1, 0) = xVq.at<uchar>(3, 0);

	Mat v21 = v2 - v1, v31 = v3 - v1, v41 = v4 - v1;
	Mat p(2, 1, CV_8UC1);
	p.at<uchar>(0, 0) = pstart_y; p.at<uchar>(1, 0) = pstart_x;
	Mat p1 = p - v1;
	double a1 = v31.at<uchar>(0, 0), a2 = v21.at<uchar>(0, 0),          //y
		a3 = v41.at<uchar>(0, 0) - v31.at<uchar>(0, 0) - v21.at<uchar>(0, 0);
	double b1 = v31.at<uchar>(1, 0), b2 = v21.at<uchar>(1, 0),      //x
		b3 = v41.at<uchar>(1, 0) - v31.at<uchar>(1, 0) - v21.at<uchar>(1, 0);
	double px = p1.at<uchar>(1, 0), py = p1.at<uchar>(0, 0);
	Mat tvec, mat_t;
	double t1n, t2n;
	double a, b, c;
	if (a3 == 0 && b3 == 0){
		hconcat(v31, v21, mat_t);
		tvec = mat_t.inv()*p1;
		t1n = tvec.at<uchar>(0, 0);
		t2n = tvec.at<uchar>(1, 0);
	}
	else{
		a = (b2*a3 - a2*b3);
		b = (-a2*b1 + b2*a1 + px*b3 - a3*py);
		c = px*b1 - py*a1;
		if (a == 0){
			t2n = -c / b;
		}
		else{
			t2n = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);
		}
		if (abs(a1 + t2n*a3) <= 0.0000001){
			t1n = (py - t2n*b2) / (b1 + t2n*b3);
		}
		else{
			t1n = (px - t2n*a2) / (a1 + t2n*a3);
		}

	}
	Mat m1 = v1 + t1n*(v3 - v1);
	Mat m4 = v2 + t1n*(v4 - v2);
	Mat ptest = m1 + t2n*(m4 - m1);

	double v1w = 1 - t1n - t2n + t1n*t2n;
	double v2w = t2n - t1n*t2n;
	double v3w = t1n - t1n*t2n;
	double v4w = t1n*t2n;

	Mat out(2, 8, CV_8UC1);
	T.at<uchar>(0, 0) = v1w;  T.at<uchar>(1, 0) = 0;
	T.at<uchar>(0, 1) = 0;    T.at<uchar>(1, 1) = v1w;
	T.at<uchar>(0, 2) = v2w;  T.at<uchar>(1, 2) = 0;
	T.at<uchar>(0, 3) = 0;    T.at<uchar>(1, 3) = v2w;
	T.at<uchar>(0, 4) = v3w;  T.at<uchar>(1, 4) = 0;
	T.at<uchar>(0, 5) = 0;    T.at<uchar>(1, 5) = v3w;
	T.at<uchar>(0, 6) = v4w;  T.at<uchar>(1, 6) = 0;
	T.at<uchar>(0, 7) = 0;    T.at<uchar>(1, 7) = v4w;

	assert(norm(T*V - p) < 0.0001);

}




void blkdiag(Mat& input1, Mat &input2, Mat& ouput){
	Mat out = Mat::zeros(input1.rows, input2.rows, CV_8UC1);
	for (int i = 0; i < input1.rows; i++){
		for (int j = 0; j < input1.cols; j++){
			out.at<uchar>(i, j) = input1.at<uchar>(i, j);
		}
	}
	for (int i = 0; i < input2.rows; i++){
		for (int j = 0; j < input2.cols; j++){
			out.at<uchar>(i + input1.rows, j + input1.cols) = input1.at<uchar>(i, j);
		}
	}
}


void reshape(Mat& yV, Mat& xV, Mat V){
	Mat out(8, 1, CV_8UC1);
	out.at<uchar>(0, 0) = xV.at<uchar>(0, 0);
	out.at<uchar>(1, 0) = yV.at<uchar>(0, 0);
	out.at<uchar>(2, 0) = xV.at<uchar>(1, 0);
	out.at<uchar>(3, 0) = yV.at<uchar>(1, 0);
	out.at<uchar>(4, 0) = xV.at<uchar>(2, 0);
	out.at<uchar>(5, 0) = yV.at<uchar>(2, 0);
	out.at<uchar>(6, 0) = xV.at<uchar>(3, 0);
	out.at<uchar>(7, 0) = yV.at<uchar>(3, 0);
	V = out;
}


void checkInterSection(Mat& lineA, Mat& lineB, int& isInterSect, Mat& p){
	Mat A1 = lineA.rowRange(1, 2).clone();
	Mat A2 = lineA.rowRange(0, 1).clone();
	Mat deltaA = A1 - A2;

	Mat B1 = lineB.rowRange(1, 2).clone();
	Mat B2 = lineB.rowRange(0, 1).clone();
	Mat deltaB = B1 - B2;

	Mat deltaA_t = deltaA.t();
	Mat deltaB_t = deltaB.t();
	Mat M;
	vconcat(-deltaA_t, deltaB_t, M);
	Mat t;
	Mat M_inv = M.inv();
	t = M_inv*((lineA.rowRange(0, 1) - lineB.rowRange(0, 1)).t());
	int t1 = 0, t0 = 0;
	int rows = t.rows;
	int cols = t.cols;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			if (t.at<uchar>(i, j) <= 1){
				t1++;
			}
			if (t.at<uchar>(i, j) >= 0){
				t0++;
			}
		}
	}
	isInterSect = false;
	if (t1 == 2 && t0 == 2){
		isInterSect = true;
	}
	p = (lineA.rowRange(0, 1) + t.at<uchar>(0, 0)*deltaA);
}

void getchannel(Mat& input_img, int n_channel, int N, Mat& output_img){
	int rows = input_img.rows;
	int cols = input_img.cols;
	Mat out(rows, cols, CV_16UC1);
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
	output_img = out;
}

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
