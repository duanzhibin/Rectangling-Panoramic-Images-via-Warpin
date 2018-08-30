#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y); //

void getvertices(int ygridID, int xgridID, Mat& yGrid, Mat& xGrid, Mat& yVq, Mat& xVq); //

void reshape(Mat& yV, Mat& xV, Mat& V);    //


void getchannel(Mat& input_img, int n_channel, int N, Mat& output_img);  //

void getchannel_int(Mat& input_img, int n_channel, int N, Mat& output_img);

void getchannel_float(Mat& input_img, int n_channel, int N, Mat& output_img);  //

void combin2Channel(Mat& inputImg1, Mat& inputImg2, Mat& outputImg);//

void combin3Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& outputImg);//

void combin4Channel(Mat& inputImg1, Mat& inputImg2, Mat& inputImg3, Mat& inputImg4, Mat& outputImg);//

void drawGridmask(Mat& ygrid,Mat& xgrid,int rows,int cols,Mat& gridmask);

void drawGrid(Mat& gridmask, Mat& image, Mat& outimage);

void ucharToFloat(Mat& inputMat);

void intToFloat(Mat& inputMat);

void floatTodouble(Mat& inputMat);

void ucharTodouble(Mat& inputMat);

void blkdiag(Mat& input1, Mat &input2, Mat& output);

void checkInterSection(Mat& lineA, Mat& lineB, int& isInter, Mat& p);

void getLinTrans(int pstart_y, int pstart_x, Mat& yVq, Mat& xVq, Mat& T);

void checkLocal(Mat& image, Mat &mask, Mat& dispMap);


int checkIsIn(Mat& yVq, Mat& xVq, int pstartx, int pstarty,int pendx,int pendy);