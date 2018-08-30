#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
class Rectangling
{

  public:
	int axs;
	Mat img, mask, xDspMap, yDispMap;
	Mat outImg, outMask, outXDispMap, outYDispMap;
	Rectangling();
	Rectangling();

	void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

	void getvertices(int ygridID, int xgridID, Mat& yGrid, Mat& xGrid, Mat& yVq, Mat& xVq);

	void getLinTrans(int pstart_y, int pstart_x, Mat& yVq, Mat& xVq, Mat& T);

	void blkdiag(Mat& input1, Mat &input2, Mat& ouput);

	void reshape(Mat& yV, Mat& xV, Mat V);

	void checkInterSection(Mat& lineA, Mat& lineB, int& isInter, Mat& p);

	void getchannel(Mat& input_img, int n_channel, int N, Mat& output_img);

	void combin2Channel(Mat& inputImg1, Mat& inputImg2, Mat& outputImg);

	void combin3Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& outputImg);

	void combin4Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& inputImg4, Mat& outputImg);
};