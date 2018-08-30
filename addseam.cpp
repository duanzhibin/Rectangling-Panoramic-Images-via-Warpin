#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include"addseam.h"
#include"myFunctions.h"
#include<math.h>
using namespace cv;
using namespace std;

#define N1 2
#define N2 3

void energy_function_x(Mat &image, Mat &output,Mat& mask){
	int addValue = 100;
	int rows = image.rows;
	int cols = image.cols;
	Mat out=Mat::zeros(rows, cols, CV_64FC1);

	for (int i = 0; i < rows; i++){
		for (int j = 1; j < cols-1; j++)
		{
			out.at<double>(i, j) = abs((int)image.at<Vec3b>(i, j + 1)[0] - (int)image.at<Vec3b>(i, j)[0]) +
				abs((int)image.at<Vec3b>(i, j + 1)[1] - (int)image.at<Vec3b>(i, j)[1]) +
				abs((int)image.at<Vec3b>(i, j + 1)[2] - (int)image.at<Vec3b>(i, j)[2]);
			/*
				abs((int)image.at<Vec3b>(i, j -1)[0] - (int)image.at<Vec3b>(i, j)[0]) +
				abs((int)image.at<Vec3b>(i, j -1)[1] - (int)image.at<Vec3b>(i, j)[1]) +
				abs((int)image.at<Vec3b>(i, j -1)[2] - (int)image.at<Vec3b>(i, j)[2]);
				*/
		}
	}

	for (int i = 1; i < rows-1; i++){
		for (int j = 0; j < cols; j++)
		{
			out.at<double>(i, j) = out.at<double>(i, j) + abs((int)image.at<Vec3b>(i + 1, j)[0] - (int)image.at<Vec3b>(i, j)[0]) +
				abs((int)image.at<Vec3b>(i + 1, j)[1] - (int)image.at<Vec3b>(i, j)[1]) +
				abs((int)image.at<Vec3b>(i + 1, j)[2] - (int)image.at<Vec3b>(i, j)[2]);
			/*
				abs((int)image.at<Vec3b>(i - 1, j)[0] - (int)image.at<Vec3b>(i, j)[0]) +
				abs((int)image.at<Vec3b>(i - 1, j)[1] - (int)image.at<Vec3b>(i, j)[1]) +
				abs((int)image.at<Vec3b>(i - 1, j)[2] - (int)image.at<Vec3b>(i, j)[2]);
				*/
		}
	}
	//Mat dx, dy;
	//Sobel(image, dx, CV_64F, 1, 0);
	//Sobel(image, dy, CV_64F, 0, 1);
	//magnitude(dx, dy, output);
	output = out.clone();
	//double min_value, max_value, Z;
	//minMaxLoc(output, &min_value, &max_value);
	//Z = 1.0 / max_value * 255;
	//output = output * Z;
	//normalize
	//output.convertTo(output, CV_8U);
	Mat x_mask = mask.clone();
	ucharTodouble(x_mask);
	output = output + 1000000*x_mask;
}

void energy_function(Mat &image, Mat &output, Mat& mask,int axs){
	Mat dx, dy;
	Mat img = image;
	ucharTodouble(img);
	Sobel(img, dx, image.depth(), 1, 0);
	Sobel(img, dy, image.depth(), 0, 1);

	ucharTodouble(dx);
	ucharTodouble(dy);
	//magnitude(dx, dy, out_1);
	Mat out = Mat::zeros(dx.rows, dx.cols, CV_64FC1); 
	//double min_value, max_value, Z;
	//minMaxLoc(output, &min_value, &max_value);
	//Z = 1 / max_value * 255;
	//output = output * Z;
	//normalize
	//output.convertTo(output, CV_64F);

#pragma omp parallel for 
	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			for (int k = 0; k < 3; k++){
				out.at <double>(i, j) += sqrt(dx.at<Vec3d>(i, j)[k]*dx.at<Vec3d>(i, j)[k]+dy.at<Vec3d>(i, j)[k] * dy.at<Vec3d>(i, j)[k]);
			}
		}
	}
	
	Mat x_mask = mask.clone();
	ucharTodouble(x_mask);

	if (axs == 1 || axs == 2){
		for (int i = 0; i < x_mask.rows; i++){
			for (int j = 0; j < x_mask.cols; j++){
				if (x_mask.at<double>(i, j) == 2){
				    x_mask.at<double>(i, j) = 1;
				}
				else if (x_mask.at<double>(i, j) == 1){
					x_mask.at<double>(i, j) = 1;
				}
				else{
				  x_mask.at<double>(i, j) = 0;
				}
			}

		}
	}
	else{

		for (int i = 0; i < x_mask.rows; i++){
			for (int j = 0; j < x_mask.cols; j++){
				if (x_mask.at<double>(i, j) == 3){
					x_mask.at<double>(i, j) = 1;
				}
				else if (x_mask.at<double>(i, j) == 1){
					x_mask.at<double>(i, j) = 1;
				}
				else{
					x_mask.at<double>(i, j) = 0;
				}
			}
		}
	}

	output = out + 10000000 * x_mask;
	//cout << output << endl;
}


int *find_seam(Mat &image){

	int H = 395;   //image.rows;
	int W = 632;   // image.cols;

	H = image.rows;
	W = image.cols;

	//int dp[395][632];                 //

	int **dp = new int*[H];
	for (int i = 0; i < H; i++)
		dp[i] = new int[W];

	for (int c = 0; c < W; c++){
		dp[0][c] = (int)image.at<double>(0, c);
	}

	for (int r = 1; r < H; r++){
		for (int c = 0; c < W; c++){
			if (c == 0)
				dp[r][c] = min(dp[r - 1][c + 1], dp[r - 1][c]);
			else if (c == W - 1)
				dp[r][c] = min(dp[r - 1][c - 1], dp[r - 1][c]);
			else
				dp[r][c] = min(min(dp[r - 1][c - 1], dp[r - 1][c]), min(dp[r - 1][c - 1], dp[r - 1][c + 1]));
			dp[r][c] += (int)image.at<double>(r, c);
		}
	}

	int min_value = 2147483647; //infinity
	int min_index = -1;
	for (int c = 0; c < W; c++)
		if (dp[H - 1][c] < min_value) {
			min_value = dp[H - 1][c];
			min_index = c;
		}

	int *path = new int[H];       //max(rows,cols); 
	Point pos(H - 1, min_index);
	path[pos.x] = pos.y;

	while (pos.x != 0){
		int value = dp[pos.x][pos.y] - (int)image.at<double>(pos.x, pos.y);
		int r = pos.x, c = pos.y;
		if (c == 0){
			if (value == dp[r - 1][c + 1])
				pos = Point(r - 1, c + 1);
			else
				pos = Point(r - 1, c);
		}
		else if (c == W - 1){
			if (value == dp[r - 1][c - 1])
				pos = Point(r - 1, c - 1);
			else
				pos = Point(r - 1, c);
		}
		else{
			if (value == dp[r - 1][c - 1])
				pos = Point(r - 1, c - 1);
			else if (value == dp[r - 1][c + 1])
				pos = Point(r - 1, c + 1);
			else
				pos = Point(r - 1, c);
		}
		path[pos.x] = pos.y;
	}
	delete(dp);
	return path;
}
void add_pixels(Mat& image, Mat& mask, Mat& xDispMap, Mat& yDispMap, Mat& output, Mat& outMask, Mat& outxDispMap, Mat& outyDispMap, int *seam, int direction, int axs){
#pragma omp parallel for
	for (int r = 0; r < image.rows; r++) {
		for (int c = 0; c < image.cols; c++){
			if (direction == 0){
				if (c == seam[r]){
					if (c < image.cols - 1){
						output.at<Vec3b>(r, c)[0] = (image.at<Vec3b>(r, c + 1)[0] + image.at<Vec3b>(r, c)[0]) / 2 + 0.5;        //
						output.at<Vec3b>(r, c)[1] = (image.at<Vec3b>(r, c + 1)[1] + image.at<Vec3b>(r, c)[1]) / 2 + 0.5;        //
						output.at<Vec3b>(r, c)[2] = (image.at<Vec3b>(r, c + 1)[2] + image.at<Vec3b>(r, c)[2]) / 2 + 0.5;        //

						if (mask.at<uchar>(r, c) == 0){
							if (axs == 1 || axs == 2){
								outMask.at<uchar>(r, c) = N1;                             //填充后的值，应该为1，这里选择与（r,c）处相等
							}
							else{
								outMask.at<uchar>(r, c) = N2;
							}
						}
						else{
							outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
						}

						outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c);
						outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c);

						// output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
						// outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
					}
					else{
						output.at<Vec3b>(r, c)[0] = (image.at<Vec3b>(r, c)[0] + image.at<Vec3b>(r, c - 1)[0]) / 2 + 0.5;        //正常情况下不会出现该情况
						output.at<Vec3b>(r, c)[1] = (image.at<Vec3b>(r, c)[1] + image.at<Vec3b>(r, c - 1)[1]) / 2 + 0.5;        //
						output.at<Vec3b>(r, c)[2] = (image.at<Vec3b>(r, c)[2] + image.at<Vec3b>(r, c - 1)[2]) / 2 + 0.5;        //

						if (mask.at<uchar>(r, c) == 0){
							if (axs == 1 || axs == 2){
								outMask.at<uchar>(r, c) = N1;                             //填充后的值，应该为1，这里选择与（r,c）处相等
							}
							else{
								outMask.at<uchar>(r, c) = N2;
							}
						}
						else{
							outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);

						}

						outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c);
						outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c);
						// output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
						// outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
					}
				}
				else if (c < seam[r]){                                            // c<seam[r]
					output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c + 1);
					outMask.at<uchar>(r, c) = mask.at<uchar>(r, c + 1);
					outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c + 1);
					outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c + 1);
				}
				else{                                                           // c>seam[r]
					output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
					outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
					outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c);
					outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c);
				}
				//outMask.at<uchar>(r, seam[r] - 1) = outMask.at<uchar>(r, seam[r] - 1) + N;
				if (mask.at<uchar>(r, c) == 0){
					if (axs == 1 || axs == 2){
						outMask.at<uchar>(r, seam[r] - 1) = N1;
					}
					else{
						outMask.at<uchar>(r, seam[r] - 1) = N2;
					}
				}
				else{
					outMask.at<uchar>(r, seam[r] - 1) = mask.at<uchar>(r, c);

				}
			}

			else{
				if (c == seam[r]){
					if (c > 0){
						output.at<Vec3b>(r, c)[0] = (image.at<Vec3b>(r, c - 1)[0] + image.at<Vec3b>(r, c)[0]) / 2 + 0.5;        //
						output.at<Vec3b>(r, c)[1] = (image.at<Vec3b>(r, c - 1)[1] + image.at<Vec3b>(r, c)[1]) / 2 + 0.5;        //
						output.at<Vec3b>(r, c)[2] = (image.at<Vec3b>(r, c - 1)[2] + image.at<Vec3b>(r, c)[2]) / 2 + 0.5;        //

						//outMask.at<uchar>(r, c) = mask.at<uchar>(r, c)+N;
						if (mask.at<uchar>(r, c) == 0){
							if (axs == 1 || axs == 2){
								outMask.at<uchar>(r, c) = N1;
							}
							else{
								outMask.at<uchar>(r, c) = N2;
							}
						}
						else{
							outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
						}


						outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c);
						outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c);
						// output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
						// outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
					}
					else{
						output.at<Vec3b>(r, c)[0] = (image.at<Vec3b>(r, c)[0] + image.at<Vec3b>(r, c + 1)[0]) / 2 + 0.5;        //正常情况下不会出现该情况
						output.at<Vec3b>(r, c)[1] = (image.at<Vec3b>(r, c)[1] + image.at<Vec3b>(r, c + 1)[1]) / 2 + 0.5;        //
						output.at<Vec3b>(r, c)[2] = (image.at<Vec3b>(r, c)[2] + image.at<Vec3b>(r, c + 1)[2]) / 2 + 0.5;        //


						//outMask.at<uchar>(r, c) = mask.at<uchar>(r, c)+N;
						if (mask.at<uchar>(r, c) == 0){
							if (axs == 1 || axs == 2){
								outMask.at<uchar>(r, c) = N1;
							}
							else{
								outMask.at<uchar>(r, c) = N2;
							}
						}
						else{
							outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);
						}


						outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c);
						outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c);
					}
				}
				else if (c < seam[r]){                                            // c<seam[r]
					output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c);
					outMask.at<uchar>(r, c) = mask.at<uchar>(r, c);

					outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c);
					outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c);
				}
				else{                                                             // c>seam[r]
					output.at<Vec3b>(r, c) = image.at<Vec3b>(r, c - 1);
					outMask.at<uchar>(r, c) = mask.at<uchar>(r, c - 1);
					outxDispMap.at<float>(r, c) = xDispMap.at<float>(r, c - 1);
					outyDispMap.at<float>(r, c) = yDispMap.at<float>(r, c - 1);
				}
				//outMask.at<uchar>(r, seam[r] + 1) = outMask.at<uchar>(r, seam[r] + 1) + N;
				if (mask.at<uchar>(r, c) == 0){
					if (axs == 1 || axs == 2){
						outMask.at<uchar>(r, seam[r] + 1) = N1;
					}
					else{
						outMask.at<uchar>(r, seam[r] + 1) = N2;
					}
				}
				else{
					outMask.at<uchar>(r, seam[r] + 1) = mask.at<uchar>(r, c);

				}

			}

		}
	}
}



void rot90(Mat &matImage, int rotflag){
	//1=CW, 2=CCW, 3=180
	if (rotflag == 1){
		transpose(matImage, matImage);
		flip(matImage, matImage, 1); //transpose+flip(1)=CW
	}
	else if (rotflag == 2) {
		transpose(matImage, matImage);
		flip(matImage, matImage, 0); //transpose+flip(0)=CCW
	}
	else if (rotflag == 3){
		flip(matImage, matImage, -1);    //flip(-1)=180
	}
	else if (rotflag != 0){ //if not 0,1,2,3:
		cout << "Unknown rotation flag(" << rotflag << ")" << endl;
	}
}

void add_seam(Mat& image, Mat& mask, Mat& xDispMap, Mat& yDispMap, char orientation, int direction,int axs){
	if (orientation == 'h'){
		rot90(image, 1);
		rot90(mask, 1);
		rot90(xDispMap, 1);
		rot90(yDispMap, 1);
	}
	//imshow("submask", mask*255);
	//waitKey(10);

	int H = image.rows, W = image.cols;

	//Mat gray;
	//cvtColor(image, gray, CV_BGR2GRAY);
	
	Mat ratioMask = mask.clone();
	if (axs == 1||axs==2){
		for (int i = 0; i < ratioMask.rows; i++){
			for (int j = 0; j < ratioMask.cols; j++){
				if (ratioMask.at<uchar>(i, j) == 2){
					ratioMask.at<uchar>(i, j) = 1;
				}
				else if (ratioMask.at<uchar>(i, j) == 1){
					ratioMask.at<uchar>(i, j) = 1;
				}
				else{
					ratioMask.at<uchar>(i, j) = 0;
				}
			}

		}
	}
	else{
		for (int i = 0; i < ratioMask.rows; i++){
			for (int j = 0; j < ratioMask.cols; j++){
				if (ratioMask.at<uchar>(i, j) == 3){
					ratioMask.at<uchar>(i, j) = 1;
				}
				else if (ratioMask.at<uchar>(i, j) == 1){
					ratioMask.at<uchar>(i, j) = 1;
				}
				else{
					ratioMask.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	Mat eimage;
	energy_function(image, eimage, mask,axs);

	int *seam = find_seam(eimage);
	//Mat output(H, W - 1, CV_8UC3);
	Mat output(H, W, CV_8UC3);          //  H*W
	Mat outMask(H, W, CV_8UC1);
	Mat outxDispMap(H, W, CV_32FC1);
	Mat outyDispMap(H, W, CV_32FC1);
	add_pixels(image,mask,xDispMap,yDispMap,output,outMask,outxDispMap,outyDispMap,seam,direction,axs);
	
	delete(seam);

	if (orientation == 'h'){
		rot90(output, 2);
		rot90(outMask, 2);
		rot90(outxDispMap, 2);
		rot90(outyDispMap, 2);
	}
	image = output.clone();
	//imshow("image", image);
	//waitKey(10);
	mask = outMask.clone();
	xDispMap = outxDispMap.clone();
	yDispMap = outyDispMap.clone();

	//imshow("outmask",mask*255);
	//waitKey(10);
}
/*/void shrink_image(Mat& image, int new_cols, int new_rows, int width, int height){
	cout << endl << "Processing image..." << endl;
	int delta_w = width - new_cols;
	int delta_h = height - new_rows;

	if (delta_w > 0){
		for (int i = 0; i < delta_w; i++){      //i<width - new_cols
			remove_seam(image, 'v');
			cout << i << endl;
		}
	}
	else if (delta_w < 0){
		delta_w = -delta_w;
		for (int i = 0; i < delta_w; i++){      //i<width - new_cols
			add_seam(image, 'v');
			cout << i << endl;
		}
	}

	if (delta_h > 0){
		for (int i = 0; i < delta_h; i++){      //i<width - new_cols
			remove_seam(image, 'h');
			cout << i << endl;
		}
	}
	else if (delta_h < 0){
		delta_h = -delta_h;
		for (int i = 0; i < delta_h; i++){      //i<width - new_cols
			add_seam(image, 'h');
			cout << i << endl;
		}
	}


}/*/


int addSeam_wrap(Mat& img, Mat& mask, Mat xDispMap, Mat yDispMap,
	Mat& outImg, Mat& outMask, Mat& outXDispMap, Mat& outYDispMap,int axs){
	if (axs == 1){                                //up
		add_seam(img,mask,xDispMap,yDispMap,'h',1,axs);
		outImg = img.clone();
		outMask = mask.clone();
		outXDispMap = xDispMap.clone();
		outYDispMap = yDispMap.clone();
	}
	else if (axs == 2){                           //down
		add_seam(img,mask,xDispMap,yDispMap,'h',0, axs);
		outImg = img.clone();
		outMask = mask.clone();
		outXDispMap = xDispMap.clone();
		outYDispMap = yDispMap.clone();
	}
	else if (axs == 3){                            //left
  		add_seam(img, mask, xDispMap, yDispMap, 'v',0,axs);
		outImg = img.clone();
		outMask = mask.clone();
		outXDispMap = xDispMap.clone();
		outYDispMap = yDispMap.clone();
	}
	else if(axs==4){                                         //right
		
	    add_seam(img, mask, xDispMap, yDispMap, 'v', 1,axs);
		outImg = img.clone();
		outMask = mask.clone();
		outXDispMap = xDispMap.clone();
		outYDispMap = yDispMap.clone();

	}
	return 0;
}
