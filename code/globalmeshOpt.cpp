#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<Eigen\Dense>
#include <opencv2/core/eigen.hpp>


#include<vector>
#include<cmath>

#include"globalmesh.h"
#include"LocalWarp.h"
#include"myFunctions.h"


using namespace std;
using namespace cv;
using namespace Eigen;

#define PI 3.14159265


void globalmeshOpt(Mat& origImg1M, Mat& mask1M, Mat& dispMap,Mat &Vlocal,Mat& Vglobal){
	//--------------    -----------   Initialization    ----------------      ------------------
	Mat image = origImg1M.clone();
	Mat mask = mask1M.clone();
	Mat u = dispMap.clone();
	
	int cols = image.cols;
	int rows = image.rows;
	Mat X, Y;
	meshgrid(Range(1, cols), Range(1, rows), X, Y);

	Mat yDispMap;
	Mat xDispMap;
	getchannel_float(dispMap, 0, 2, yDispMap);
	getchannel_float(dispMap, 1, 2, xDispMap);              //正确

	// Define grid mesh V
	int xgridN =30;                                     //30*20
	int ygridN =20;
	Mat xgrid(ygridN, xgridN, CV_32FC1), ygrid(ygridN,xgridN,CV_32FC1); 
	//meshgrid(Range(1,cols),Range(1,rows),xgrid,ygrid);  
	int x = 0, y = 0;
	for (double i = 0; i < rows; i = i + double((rows - 1)) / (ygridN - 1)){
		for (double j = 0; j < cols; j = j + double((cols - 1)) / (xgridN - 1)){
			ygrid.at<float>(y, x) = int(i);
			xgrid.at<float>(y, x) = int(j);
			//cout << int(i) << " "<<int(j) << endl;
			x++;
		}
		y++;
		x = 0;
	}
	Mat yVopt = Mat::zeros(ygridN, xgridN, CV_32FC1);
	Mat xVopt = Mat::zeros(ygridN, xgridN, CV_32FC1);
	// Draw grid on image
	
	//cout << ygrid << xgrid << endl;
	
	Mat gridmask,imageGrided;
	drawGridmask(ygrid,xgrid, rows, cols, gridmask);
	drawGrid(gridmask,image,imageGrided);
	imshow("imageGrided", imageGrided);
	waitKey(30);
	
	//-----------------Wrap grid according to displacement field-----------------
	int yN = xgrid.rows;
	int xN = xgrid.cols;
	Mat xVwarp(yN, xN, CV_32FC1);
	Mat yVwarp(yN, xN, CV_32FC1);
	for (int y = 0; y < yN; y++){
		for (int x = 0; x < xN; x++){
			yVwarp.at<float>(y, x) = ygrid.at<float>(y, x) +
				yDispMap.at<float>(ygrid.at<float>(y, x), xgrid.at<float>(y, x));

			xVwarp.at<float>(y, x) = xgrid.at<float>(y, x) +
				xDispMap.at<float>(ygrid.at<float>(y, x), xgrid.at<float>(y, x));
			
		}
	}
	//cout << yVwarp.type() << "\n" << xVwarp.type();
	
	// Draw grid on image
	
	Mat gridmask2, imageGrided2;
	drawGridmask(yVwarp, xVwarp,rows, cols, gridmask2);
	drawGrid(gridmask2, image, imageGrided2);
	Mat imgx = imageGrided2.clone();
	imshow("imageGrided2", imageGrided2);
	waitKey(30);
	
	//以上正确   2018/08/05  15：48


	//--------------------Shape preservation matrix  ----------------
	int quadrows = ygridN - 1;
	int quadcols = xgridN - 1;

	Mat Vo(4, 2, CV_32FC1);
	Mat Aq(8, 4, CV_32FC1);
	Mat **Ses = new Mat*[quadrows];
	for (int i = 0; i < quadrows; i++){
		Ses[i] = new Mat[quadcols];
	}
	Mat **lineSeg = new Mat*[quadrows];
	for (int i = 0; i < quadrows; i++){
		lineSeg[i] = new Mat[quadcols];
	}
	Mat **linegroup = new Mat*[quadrows];
	for (int i = 0; i < quadrows; i++){
		linegroup[i] = new Mat[quadcols];
	}
	// optimization loop
	Mat **Cmatrixes = new Mat*[quadrows];
	for (int i = 0; i < quadrows; i++){
		Cmatrixes[i] = new Mat[quadcols];
	}
	Mat **LinTrans = new Mat*[quadrows];
	for (int i = 0; i < quadrows; i++){
		LinTrans[i] = new Mat[quadcols];
	}

	for (int r = 0; r < quadrows; r++){
		for (int c = 0; c < quadcols; c++){
			Vo.at<float>(0, 0) = yVwarp.at<float>(r, c);
		    Vo.at<float>(0, 1) = xVwarp.at<float>(r, c);
			Vo.at<float>(1, 0) = yVwarp.at<float>(r, c+1);
			Vo.at<float>(1, 1) = xVwarp.at<float>(r, c+1);
			Vo.at<float>(2, 0) = yVwarp.at<float>(r+1, c);
			Vo.at<float>(2, 1) = xVwarp.at<float>(r+1, c);
			Vo.at<float>(3, 0) = yVwarp.at<float>(r+1, c+1);
			Vo.at<float>(3, 1) = xVwarp.at<float>(r+1, c+1);

			for (int i = 0; i < 4; i++){
				Aq.at<float>(i*2,0) = Vo.at<float>(i, 1);
				Aq.at<float>(i*2,1) = -Vo.at<float>(i, 0);
				Aq.at<float>(i*2+1,0) = Vo.at<float>(i, 0);
				Aq.at<float>(i*2+1,1) = Vo.at<float>(i, 1);
				Aq.at<float>(i * 2 , 2) = 1;
				Aq.at<float>(i * 2 , 3) = 0;
				Aq.at<float>(i * 2 + 1, 2) = 0;
				Aq.at<float>(i * 2 + 1, 3) = 1;
			}
			//cout << Aq << endl;

			Mat eye_8 = Mat::eye(cv::Size(8, 8), CV_32FC1);
			Mat Aq_t;
			Aq_t = Aq.t();
			Ses[r][c] = Aq *(Aq_t*Aq).inv()*Aq_t-eye_8 ;
			//cout << Ses[r][c]<<endl;
		}
	}
	//--------------------boundary constraints matrix ----------------------------
	int vertexesNum = xgridN*ygridN;
	Mat Dvec = Mat::zeros(vertexesNum * 2, 1, CV_32FC1);
	Mat B = Mat::zeros(vertexesNum * 2, 1, CV_32FC1);
	for (int i = 0; i < vertexesNum*2; i = i + xgridN * 2){
		Dvec.at < float>(i, 0) = 1;
		B.at < float >(i, 0) = 1;
	}
	for(int i = xgridN * 2 - 1-1;i<vertexesNum*2;i=i+xgridN*2){
		Dvec.at<float>(i,0) = 1;
		B.at<float>(i, 0) = cols;
	}
	for (int i =2-1; i<xgridN*2; i = i + 2){
		Dvec.at<float>(i, 0) = 1;
		B.at<float>(i, 0) =1;
	}
	for (int i = vertexesNum*2-xgridN*2+2-1; i<vertexesNum*2; i = i + 2){
		Dvec.at<float>(i, 0) = 1;
		B.at<float>(i, 0) = rows;
	}
	//cout << Dvec << endl;
	//cout << B << endl;

	Mat Dg = Mat::diag(Dvec);
	//cout << Dg << endl;

	//以上正确  2018/8/5 16：49
	// ------------    ---------------    line preservation     ------------       -----------------    ---------------------

	Mat img_gray;
	cvtColor(image, img_gray, CV_BGR2GRAY);
	Mat line_gray = img_gray.clone();
	vector<Vec4f> lines;
	// Create and LSD detector with standard or no refinement.
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	//Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
	ls->detect(img_gray, lines);
	//显示图片线条
	Mat drawnLines(img_gray);
	ls->drawSegments(drawnLines, lines);
	imshow("Standard refinement", drawnLines);
	waitKey(30);

	Mat fatmask = mask.clone();
	int end_col = mask.cols;
	int end_row = mask.rows;
	for (int i = 0; i < end_row; i++){
		fatmask.at<uchar>(i, 0) = 1;
		fatmask.at<uchar>(i, end_col - 1) = 1;
	}
	for (int j = 0; j < end_row; j++){
		fatmask.at<uchar>(0, j) = 1;
		fatmask.at<uchar>(end_row-1, j) = 1;
	}
	filter2D(fatmask, fatmask, fatmask.depth(),Mat::ones(3,3,CV_8UC1));
	//cout << fatmask << endl;
	//-------------cut the lines by grid------------------------
	int lineN = lines.size();
	//build warping from input to output
	Mat y_iu(rows, cols, CV_32FC1);
	Mat x_iu(rows, cols, CV_32FC1);


	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			y_iu.at<float>(int(i + yDispMap.at<float>(i, j)), int(j + xDispMap.at < float >(i, j))) = i;
			x_iu.at<float>(int(i + yDispMap.at<float>(i, j)), 
				int(j + xDispMap.at < float >(i, j))) = j;
		}
	}

	Mat aLine(2, 2, CV_32SC1);
	int line1out_y, line1out_x;
	int line2out_y, line2out_x;
	double gridwidth, gridheigth;
	int line1gridID_x, line1gridID_y, line2gridID_x, line2gridID_y;
	int pstart_y, pstart_x, pend_y, pend_x;
	int currentGrid_y, currentGrid_x;
	Mat gridstep(2, 4, CV_32SC1);
	int lineplot1 = 0, lineplot2 = 0;

	int **line_flag = new int*[quadrows];
	for (int i = 0; i < quadrows; i++){
		line_flag[i] = new int[quadcols];
	}
	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			line_flag[i][j] = 0;
		}
	}

	for (int i = 0; i < lineN; i++){   //应该是lineN
		// lines  x1,y1,x2,y2
		aLine.at<int>(0, 0) = int(lines[i][1]); aLine.at<int>(0, 1) = int(lines[i][0]);   // y1 x1
		aLine.at<int>(1, 0) = int(lines[i][3]); aLine.at<int>(1, 1) = int(lines[i][2]);   // y2 x2
		//cout << aLine << endl;
		// aline might contain negative coordinate
		if (fatmask.at<uchar>(aLine.at<int>(0, 0), aLine.at<int>(0, 1))>0 ||
			fatmask.at<uchar>(aLine.at<int>(1, 0), aLine.at<int>(1, 1)) > 0){
			continue;
		}
		//cout <<int( aLine.at<uchar>(0, 0)) << " " <<int( aLine.at<uchar>(0, 1)) << endl;

		line1out_y = y_iu.at<float>(int(aLine.at<int>(0, 0)), int(aLine.at<int>(0, 1)));
		line1out_x = x_iu.at<float>(int(aLine.at<int>(0, 0)), int(aLine.at<int>(0, 1)));

		//cout << line1out_y << line1out_x;

		line2out_y = y_iu.at<float>(int(aLine.at<int>(1, 0)), int(aLine.at<int>(1, 1)));
		line2out_x = x_iu.at<float>(int(aLine.at<int>(1, 0)), int(aLine.at<int>(1, 1)));

		gridwidth = double(cols - 1) / (xgridN - 1);
		gridheigth = double(rows - 1) / (ygridN - 1);


		line1gridID_y = int(double(line1out_y) / gridheigth + 1 - 1);
		line1gridID_x = int(double(line1out_x) / gridwidth + 1 - 1);
		line2gridID_y = int(double(line2out_y) / gridheigth + 1 - 1);
		line2gridID_x = int(double(line2out_x) / gridwidth + 1 - 1);


		//cout << line1out_y << "   " << line1out_x << "   " << line2out_y << "   " << line2out_x << endl;                 //正确
		//cout << line1gridID_y << "   " << line1gridID_x << "   " << line2gridID_y << "   " << line2gridID_x << endl;     //正确

		combin3Channel(line_gray, line_gray, line_gray, line_gray);
		
		//line(imgx, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, CV_AA);       //动态显示直线切过程
		//imshow("line", imgx);
		//waitKey(30);

		currentGrid_y = line1gridID_y;
		currentGrid_x = line1gridID_x;

		pstart_y = aLine.at<int>(0, 0), pstart_x = aLine.at<int>(0, 1);
		pend_y = aLine.at<int>(1, 0), pend_x = aLine.at<int>(1, 1);

		gridstep.at<int>(0, 0) = 0;  gridstep.at<int>(1, 0) = -1;     //第一个对应y,第二个对应x
		gridstep.at<int>(0, 1) = -1; gridstep.at<int>(1, 1) = 0;
		gridstep.at<int>(0, 2) = 1;  gridstep.at<int>(1, 2) = 0;
		gridstep.at<int>(0, 3) = 0;  gridstep.at<int>(1, 3) = 1;

		//cout << pstart_x << pstart_y;

		Mat pstart(1, 2, CV_32FC1);
		pstart.at<float>(0, 0) = pstart_y;
		pstart.at<float>(0, 1) = pstart_x;

		Mat pend(1, 2, CV_32FC1);
		pend.at<float>(0, 0) = pend_y;
		pend.at<float>(0, 1) = pend_x;

		Mat quadlines = Mat::zeros(2, 2, CV_32SC4);
		Mat xVq(4, 1, CV_32SC1), yVq(4, 1, CV_32SC1);
		int findInterSect;
		int isInter = 0;
		Mat p;
		Mat mat_t;
		Mat tt_zero(1, 1, CV_32FC1); tt_zero.at<float>(0, 0) = 0;

		Mat aLine_2;
		aLine_2 = aLine.rowRange(1, 2);
	    aLine_2.convertTo(aLine_2, CV_32F);

		Mat quadlines_1;
		int nextGrid_y, nextGrid_x;
		int lineplot1 = 0; lineplot2 = 0;
		while (1){
			if (currentGrid_x == line2gridID_x&&currentGrid_y == line2gridID_y){
				getvertices(currentGrid_y, currentGrid_x, yVwarp, xVwarp, yVq, xVq);

				pend_y = aLine.at<int>(1, 0);
				pend_x = aLine.at<int>(1, 1);

				pend.at<float>(0, 0) = pend_y;
				pend.at<float>(0, 1) = pend_x;
				
				if (pstart.type() == CV_32SC1){
					intToFloat(pstart);
				}

				if (!checkIsIn(yVq, xVq, int(pstart.at<float>(0, 1)), int(pstart.at<float>(0, 0)),
					int(pend.at<float>(0, 1)), int(pend.at<float>(0, 0)))){
					break;
				}
			}
			else{
				getvertices(currentGrid_y, currentGrid_x, yVwarp, xVwarp, yVq, xVq);
				//cout << yVq << "\n" << xVq;
				quadlines.at<Vec4i>(0, 0)[0] = yVq.at<int>(0, 0); quadlines.at<Vec4i>(0, 1)[0] = xVq.at<int>(0, 0);
				quadlines.at<Vec4i>(1, 0)[0] = yVq.at<int>(2, 0); quadlines.at<Vec4i>(1, 1)[0] = xVq.at<int>(2, 0);

				quadlines.at<Vec4i>(0, 0)[1] = yVq.at<int>(0, 0); quadlines.at<Vec4i>(0, 1)[1] = xVq.at<int>(0, 0);
				quadlines.at<Vec4i>(1, 0)[1] = yVq.at<int>(1, 0); quadlines.at<Vec4i>(1, 1)[1] = xVq.at<int>(1, 0);

				quadlines.at<Vec4i>(0, 0)[2] = yVq.at<int>(2, 0); quadlines.at<Vec4i>(0, 1)[2] = xVq.at<int>(2, 0);
				quadlines.at<Vec4i>(1, 0)[2] = yVq.at<int>(3, 0); quadlines.at<Vec4i>(1, 1)[2] = xVq.at<int>(3, 0);

				quadlines.at<Vec4i>(0, 0)[3] = yVq.at<int>(1, 0); quadlines.at<Vec4i>(0, 1)[3] = xVq.at<int>(1, 0);
				quadlines.at<Vec4i>(1, 0)[3] = yVq.at<int>(3, 0); quadlines.at<Vec4i>(1, 1)[3] = xVq.at<int>(3, 0);

				//cout << quadlines << endl;

				findInterSect = false;
				cv::vconcat(pstart, aLine_2, mat_t);
				if (mat_t.type() == CV_32SC1){
					intToFloat(mat_t);
				}
				//cout << pstart << " aLine2  "<<aLine_2 << endl;               //显示起止点
				for (int l = 0; l < 4; l++){
					getchannel_int(quadlines, l, 4, quadlines_1);
					//cout << quadlines_1 << mat_t << endl;
					/*
					circle(imgx, Point(quadlines_1.at<float>(0, 1), quadlines_1.at<float>(0, 0)), 3, Scalar(0, 0, 255), -1);
					circle(imgx, Point(quadlines_1.at<float>(1, 1), quadlines_1.at<float>(1, 0)), 3, Scalar(0, 0, 255), -1);
					imshow("line", imgx);
					waitKey(30);
					circle(imgx, Point(mat_t.at<float>(0, 1), mat_t.at<float>(0, 0)), 3, Scalar(0, 0, 255), -1);
					circle(imgx, Point(mat_t.at<float>(1, 1), mat_t.at<float>(1, 0)), 3, Scalar(0, 0, 255), -1);
					imshow("line", imgx);
					waitKey(30);*/
					checkInterSection(quadlines_1, mat_t, isInter, p);                       //此处可能有问题，应该是p出问题,两个mat指向了同一个地址
					if (pstart.type() == CV_32SC1){
						intToFloat(pstart);
					}
					/*
					cout << isInter << endl;
					if (isInter){
						pstart.at<float>(0, 0) = mat_t.at<float>(0, 0);
						pstart.at<float>(0, 1) = mat_t.at<float>(0, 1);
						cout << pstart << endl;
						cout << "P" << p << "  " << norm(pstart, p, NORM_L2) << endl;

					}*/

					if (isInter&&norm(pstart - p) > 0){
						pend = p.clone();
						nextGrid_y = currentGrid_y + gridstep.at<int>(0, l);   //
						nextGrid_x = currentGrid_x + gridstep.at<int>(1, l);   //
						findInterSect = true;
					}
				}
				//cout << "\n";
				//cout << nextGrid_y <<"  "<< nextGrid_x << endl;
				if (findInterSect == false){
					//cout << " false" << endl;
					break;

				}
				if (!checkIsIn(yVq, xVq,int(pstart.at<float>(0,1)), int(pstart.at<float>(0,0)),
					int(pend.at<float>(0,1)),int(pend.at<float>(0,0)))){
					break;
				}
			}
			//cout << pend.at<float>(0, 1) << "   " << pend.at<float>(0, 0) << endl;
			//circle(imgx, Point(pend.at<float>(0,1),pend.at<float>(0,0)), 3, Scalar(0, 0, 255), -1);    //可动态显示直线划分过程
			//imshow("line", imgx);
			//waitKey(300);

			hconcat(pstart, pend, mat_t);
			hconcat(mat_t, tt_zero, mat_t);

			//if (currentGrid_y<0 || currentGrid_y >=ygridN-1||currentGrid_x<0||currentGrid_x>=xgridN-1){
			//	break;
			//}
			
			if (line_flag[currentGrid_y][currentGrid_x] == 0){
				lineSeg[currentGrid_y][currentGrid_x] = mat_t.clone();
				line_flag[currentGrid_y][currentGrid_x]++;
			}
			else{
				cv::vconcat(lineSeg[currentGrid_y][currentGrid_x], mat_t, lineSeg[currentGrid_y][currentGrid_x]);
			}
		
			//cout << currentGrid_y <<"  " <<line2gridID_y <<" "<< currentGrid_x <<"  "<< line2gridID_x << endl;
			//cout << nextGrid_y << "  " << nextGrid_x << endl;                                           //正确

			//cout << lineSeg[currentGrid_y][currentGrid_x] << endl;

			if (currentGrid_y == line2gridID_y && currentGrid_x == line2gridID_x){
				break;
			}
			currentGrid_y = nextGrid_y;
			currentGrid_x = nextGrid_x;

			pstart = pend.clone();

			Mat aline_bx=aLine_2.clone();
			intToFloat(aline_bx);
			if (norm(pstart - aline_bx) < 1){
				break;
			}
		}

	}
	delete(line_flag);


	//---------------------------------Build shape matrix--------------------
	quadrows = ygridN - 1;
    quadcols = xgridN - 1;
	int quadID;
	int topleftverterID;
	Mat Q = Mat::zeros(8*quadrows*quadcols, 2*ygridN*xgridN, CV_8UC1);
	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			quadID = (i*quadcols + j) * 8;
			topleftverterID = (i*xgridN + j) * 2;

			Q.at<uchar>(quadID, topleftverterID) = 1; Q.at<uchar>(quadID, topleftverterID + 1) = 0;
			Q.at<uchar>(quadID + 1, topleftverterID) = 0; Q.at<uchar>(quadID + 1, topleftverterID+1) = 1;

			Q.at<uchar>(quadID + 2, topleftverterID + 2) = 1; Q.at<uchar>(quadID + 2, topleftverterID + 3) = 0;
			Q.at<uchar>(quadID + 3, topleftverterID + 2) = 0; Q.at<uchar>(quadID + 3, topleftverterID + 3) = 1;

			Q.at<uchar>(quadID + 4, topleftverterID + xgridN * 2) = 1; Q.at<uchar>(quadID + 4, topleftverterID + xgridN * 2 + 1) = 0;
			Q.at<uchar>(quadID + 5, topleftverterID + xgridN * 2) = 0; Q.at<uchar>(quadID + 5, topleftverterID + xgridN * 2 + 1) = 1;

			Q.at<uchar>(quadID + 6, topleftverterID + xgridN * 2 + 2) = 1; Q.at<uchar>(quadID + 6, topleftverterID + xgridN * 2 + 3) = 0;
			Q.at<uchar>(quadID + 7, topleftverterID + xgridN * 2 + 2) = 0; Q.at<uchar>(quadID + 7, topleftverterID + xgridN * 2 + 3) = 1;
		}
	}
//	cout << Q << endl;
	Mat S;
	int S_flag = 0;
	int Si_flag = 0;
	for (int i = 0; i < quadrows; i++){
		Mat Si;
		Si_flag = 0;
		for (int j = 0; j < quadcols; j++){
			if (Si_flag == 0){
				Si = Ses[i][j];
				Si_flag++;
			}
			else{
				blkdiag(Si, Ses[i][j], Si);
			}
		}
		if (S_flag == 0){
			S = Si;
			S_flag++;
		}
		else{
			blkdiag(S, Si, S);
		}
	}
	//cout << S << endl;

	//2018/8/16  以上正确
	//------------calculate theta of original line--------------------------
	int **linegroup_flag = new int*[quadrows];
	for (int i = 0; i < quadrows; i++){
		linegroup_flag[i] = new int[quadcols];
	}
	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			linegroup_flag[i][j] = 0;
		}
	}

	//int linegroup_flag = 0;
	int lineNum;
	Mat lineSeg_ij;
	double qstep = PI / 49;
	double theta;
	int groupid;
	Mat Mat_down(1, 2, CV_32FC1);


	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			lineSeg_ij = lineSeg[i][j];
			lineNum = lineSeg_ij.rows;   //待定
			
			for (int l = 0; l < lineNum; l++){
				pstart_y = lineSeg_ij.at<float>(l,0); pstart_x = lineSeg_ij.at<float>(l,1);
				pend_y = lineSeg_ij.at<float>(l,2); pend_x = lineSeg_ij.at<float>(l,3);

				//line(imgx, Point(pstart_x, pstart_y), Point(pend_x, pend_y), Scalar(0, 0, 255), 1, CV_AA);       //动态显示直线切过程
				//imshow("line", imgx);
				//waitKey(30);

				if (pstart_x == pend_x){
					theta = PI / 2;
				}
				else{
			    theta = atan(double(pstart_y - pend_y) / (pstart_x - pend_x));
				}

				groupid = int(double(theta + PI / 2) / qstep) ;

				//cout << theta << "  "<<groupid<<endl;
				Mat_down.at<float>(0, 0) = groupid;
				Mat_down.at<float>(0, 1) = theta;

				//cout << Mat_down.at<float>(0, 1) << endl;
				if (linegroup_flag[i][j] == 0){
					linegroup[i][j] = Mat_down.clone();
					linegroup_flag[i][j]++;
				}
				else{
				cv::vconcat(linegroup[i][j], Mat_down, linegroup[i][j]);
				}
			}
			//cout << linegroup[i][j] << endl;
		}
	}
	delete(linegroup_flag);

	//--------------------    ---------------    optimization loop    ------------------     -------------------------------
	int itNum = 10;
	int **LinTrans_flag = new int*[quadrows];
	for (int i = 0; i < quadrows; i++){
		LinTrans_flag[i] = new int[quadcols];
	}
	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			LinTrans_flag[i][j] = 0;
		}
	}

	int **Cmatrixes_flag = new int*[quadrows];
	for (int i = 0; i < quadrows; i++){
		Cmatrixes_flag[i] = new int[quadcols];
	}
	for (int i = 0; i < quadrows; i++){
		for (int j = 0; j < quadcols; j++){
			Cmatrixes_flag[i][j] = 0;
		}
	}
	Mat xVq(4, 1, CV_32SC1), yVq(4, 1, CV_32SC1);
	ucharToFloat(Q);


	for (int it = 0; it < itNum; it++)
	{
		cout << it << endl;
		for (int i = 0; i < quadrows; i++){
			for (int j = 0; j < quadcols; j++){
				lineNum = lineSeg[i][j].rows;

				for (int l = 0; l < lineNum; l++){
					getvertices(i, j, yVwarp, xVwarp, yVq, xVq);  // yVq(int)  xVq(int) yVarp(float) xVarp(float)

					pstart_y = lineSeg[i][j].at<float>(l, 0);     //
					pstart_x = lineSeg[i][j].at<float>(l, 1);
					pend_y = lineSeg[i][j].at<float>(l, 2);
					pend_x = lineSeg[i][j].at<float>(l, 3);


					//line(imgx, Point(pstart_x, pstart_y), Point(pend_x, pend_y), Scalar(0, 0, 255), 1, CV_AA);       //动态显示直线切过程
					//imshow("line", imgx);
					//waitKey(300);

					//cout << pstart_y << "  " << pstart_x << endl;
					//cout << pend_y << "  " << pend_x << endl;
					Mat CT = Mat::zeros(2, 8, CV_32FC1);
					if (checkIsIn(yVq,xVq,pstart_x,pstart_y,pend_x, pend_y))
					{

						Mat T1, T2, T;
						getLinTrans(pstart_y, pstart_x, yVq, xVq, T1);   //T1(float)
						getLinTrans(pend_y, pend_x, yVq, xVq, T2);      //T2(float)   getLinTrans经检验正确

						hconcat(T1, T2, T);                            //T(float)


						//cout << T1 << T2 << endl;
						if (LinTrans_flag[i][j] == 0){
							LinTrans[i][j] = T;
							LinTrans_flag[i][j]++;
						}
						else{
							cv::vconcat(LinTrans[i][j], T, LinTrans[i][j]);
						}

						Mat ehat(2, 1, CV_32FC1);
						ehat.at<float>(0, 0) = pstart_x - pend_x;
						ehat.at<float>(1, 0) = pstart_y - pend_y;
						theta = lineSeg[i][j].at<float>(0, 4);
						Mat R(2, 2, CV_32FC1);
						R.at<float>(0, 0) = cos(theta); R.at<float>(0, 1) = -sin(theta);
						R.at<float>(1, 0) = sin(theta); R.at<float>(1, 1) = cos(theta);
						Mat ehat_t, R_t;

						ehat_t = ehat.t();
						R_t = R.t();

						Mat eye_2 = Mat::eye(2, 2, CV_32FC1);

						Mat C = R*ehat*(ehat_t*ehat).inv()*ehat_t*R_t - eye_2;
						//cout << C << endl;
						Mat CT = C*(T1 - T2);

						//cout << T1 << " \n" << T2 << endl;
						//cout << C << endl;
						//cout << CT << endl;
					}
					else{
						cout << 1<<endl;

					}

					if (Cmatrixes_flag[i][j] == 0){
						Cmatrixes[i][j] = CT;
						Cmatrixes_flag[i][j]++;
					}
					else{
						cv::vconcat(Cmatrixes[i][j], CT, Cmatrixes[i][j]);  //CT
					}
					//cout << i << "  " << j << endl;

					//cout << Cmatrixes[i][j] << endl;
				}
			}
		}
		//update line matrix includes H (quads to lines)
		
		//判断L正确性


		Mat L;
		int L_flag = 0, Li_flag = 0;
		int N1 = 0;
		int n,m=0;
		int lineNum;

		for (int i = 0; i < quadrows; i++){
			Li_flag = 0;
			n = 0;
			Mat Li;
			for (int j = 0; j < quadcols; j++){
				//cout << i << "  " << j << endl;
			   // cout << Cmatrixes[i][j] << endl;                //Cmatrixes[i][j] 出错

				lineNum = lineSeg[i][j].rows;
				N1 = N1 + lineNum;

				if (lineNum == 0){
					if (Li_flag != 0){
						Mat x = Mat::zeros(Li.rows, 8, CV_32FC1);
						cv::hconcat(Li,x, Li);
					}
					else{
						n = n + 8;
					}
				}
				else{
					if (Li_flag == 0){
						if (n != 0){
              				Li = Mat::zeros(Cmatrixes[i][j].rows, n, CV_32FC1);
							cv::hconcat(Li, Cmatrixes[i][j], Li);
						}
						else{
							Li = Cmatrixes[i][j].clone();
						}
						Li_flag++;
					}
					else{
						blkdiag(Li, Cmatrixes[i][j], Li);
					}
				}
			}

			//cout << Li << endl;
			if (L_flag == 0&&Li_flag==0){
				m = m + n;
			}
			else if (L_flag == 0 && Li_flag != 0){
				if (m != 0){
					L = Mat::zeros(Li.rows, m, CV_32FC1);
					cv::hconcat(L, Li, L);
				}
				else{
					L = Li;
				}
				L_flag++;
			}
			else{
				if (Li_flag==0){ 
					Li = Mat::zeros(L.rows, n, CV_32FC1);
					cv::hconcat(L, Li, L);
				}
				else{
					blkdiag(L, Li, L);
				}
			}
			
		}
                                 //L出错  2018/8/7    0;55  改正错误
		//
		
		
		//cout << L.row(1) << endl;              //L出错  2018/8/7    2：29
		//FileStorage fs(".\\L.txt", FileStorage::WRITE);
		//fs << "L" << L;   // << L << S << K << BA;
		//fs.release();

		//combine matrixes
		double Nq = quadrows*quadcols;                   //当lambl=0时，正确，问题应该出现在L上面。
		double lambl = 100;
	    double lambB = 1e8;
		
		Mat BA;
	
		
		//cout << N1 << endl;

		MatrixXd S_matrix, Q_matrix, L_matrix,Dg_matrix;
		MatrixXd x1_matrix, x2_matrix, x3_matrix;

		cv2eigen(S, S_matrix);
		cv2eigen(Q, Q_matrix);
		cv2eigen(L, L_matrix);
		cv2eigen(Dg, Dg_matrix);

		x1_matrix = (1.0/Nq)*S_matrix*Q_matrix;            //    S,    Q     x1正确
	    x2_matrix = (lambl/N1)*L_matrix*Q_matrix;              //L    x2出错  2018/8/7   2：27
	    x3_matrix = lambB*Dg_matrix;
		

		Mat x1, x2, x3;
		eigen2cv(x1_matrix, x1);
		eigen2cv(x2_matrix, x2);
		eigen2cv(x3_matrix, x3);

		//cout << x1 << endl;
		//cout << x1<< endl;

		Mat K;
		vconcat(x1, x2,K);
		vconcat(K, x3, K);

		cv::vconcat(Mat::zeros(K.rows - B.rows, 1, CV_32FC1), lambB*B, BA);
		//FileStorage fs(".\\BA.txt", FileStorage::WRITE);
		//fs << "BA" << Q;   // << L << S << K << BA;
		//fs.release();

		//Update V solving linear system
		//cout << BA.type() << endl;

	    //floatTodouble(K);
		//floatTodouble(BA);

		MatrixXd  K_matrix,BA_matrix,A_matrix, b_matrix;

		cv2eigen(K,K_matrix);
		cv2eigen(BA, BA_matrix);

		A_matrix = K_matrix.transpose()*K_matrix;                            //BA   正确   K可能出错
		b_matrix = K_matrix.transpose()*BA_matrix;

		MatrixXd Ainv_matrix = A_matrix.inverse();
		MatrixXd x_matrix = Ainv_matrix*b_matrix;                         //--------------此处出现严重错误-----------------------//
	
		
		Mat x;
		eigen2cv(x_matrix, x);
		//x = x - 1;
		//cout <<x<< endl;
		//cout << x.type() << endl;
		int xid;
 		for (int i = 0; i < ygridN; i++){
			for (int j = 0; j < xgridN; j++){
				xid = (i*xgridN + j) * 2;
				xVopt.at<float>(i, j)=x.at<double>(xid,0)-1;         //
				yVopt.at<float>(i, j)=x.at<double>(xid+1,0)-1;         //
				//cout << yVopt.at<float>(i, j) << "  "<<xVopt.at<float>(i, j) << endl;
			}
		}
		
		//calculate the angle of new line segs

		Mat thetagroup = Mat::zeros(50, 1, CV_32FC1);
		Mat thetagroupNum = Mat::zeros(50, 1, CV_32FC1);

		for (int i = 0; i < quadrows; i++){
			for (int j = 0; j < quadcols; j++){
				lineNum = lineSeg[i][j].rows;
				getvertices(i, j, yVopt, xVopt, yVq, xVq);

				for (int l = 0; l < lineNum; l++){
					Mat T1, T2;
					T1 = LinTrans[i][j](Rect(0,l*2,8,2));
					T2 = LinTrans[i][j](Rect(8,l*2,8,2));

					Mat Vq(yVq.rows*2,yVq.cols,CV_32FC1);
					Vq.at<float>(0, 0) = xVq.at<float>(0, 0); Vq.at<float>(1, 0) = yVq.at<float>(0, 0);
					Vq.at<float>(2, 0) = xVq.at<float>(1, 0); Vq.at<float>(3, 0) = yVq.at<float>(1, 0);
					Vq.at<float>(4, 0) = xVq.at<float>(2, 0); Vq.at<float>(5, 0) = yVq.at<float>(2, 0);
					Vq.at<float>(6, 0) = xVq.at<float>(3, 0); Vq.at<float>(7, 0) = yVq.at<float>(3, 0);

					Mat pstartnew=T1*Vq;
				    Mat pendnew=T2*Vq;
					double theta = atan((pstartnew.at<float>(1,0)-pendnew.at<float>(1,0))/
						(pstartnew.at<float>(0, 0) - pendnew.at<float>(0, 0)));

					double deltatheta=theta-linegroup[i][j].at<float>(l,1);

					if(isnan(deltatheta)||isnan(double(linegroup[i][j].at<float>(0,1)))){
						continue;
					}
					if (deltatheta > PI / 2){
						deltatheta = deltatheta - PI;
					}
					if (deltatheta < -PI / 2){
						deltatheta = deltatheta + PI;
					}
					thetagroup.at<float>(linegroup[i][j].at<float>(l, 0), 0) =
						thetagroup.at<float>(linegroup[i][j].at<float>(l, 0), 0) +
						deltatheta;
					thetagroupNum.at<float>(linegroup[i][j].at<float>(l, 0), 0) =
						thetagroupNum.at<float>(linegroup[i][j].at<float>(l, 0
						)) + 1;
				}

			}
		}

		//calculate mean theta of each bin
		divide(thetagroup, thetagroupNum, thetagroup);

		for (int i = 0; i < quadrows; i++){
			for (int j = 0; j < quadcols; j++){
				lineNum = lineSeg[i][j].rows;
				//cout <<linegroup[i][j] << endl;
				for (int l = 0; l < lineNum; l++){
					//cout << linegroup[i][j].at<float>(l, 0) << endl;
					lineSeg[i][j].at<float>(l, 4)=thetagroup.at<float>(linegroup[i][j].at<float>(l,0),0);
				}
				
			}
		}        
	}

	delete(Ses);
	delete(lineSeg);
	delete(linegroup);
	delete(Cmatrixes);
	delete(LinTrans);
	delete(Cmatrixes_flag);
	delete(LinTrans_flag);

	Mat xVlocal = xVwarp.clone(), yVlocal = yVwarp.clone();
	Mat xVglobal = xVopt.clone(),yVglobal = yVopt.clone();
	Mat Vlocal_out, Vglobal_out;
	combin2Channel(yVlocal, xVlocal, Vlocal_out);
	combin2Channel(yVglobal,xVglobal, Vglobal_out);

	Vlocal = Vlocal_out.clone();
	Vglobal = Vglobal_out.clone();

}

/*
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
	yVq.at<uchar>(1, 0) = yGrid.at<uchar>(ygridID, xgridID+1);
	yVq.at<uchar>(2, 0) = yGrid.at<uchar>(ygridID+1, xgridID);
	yVq.at<uchar>(3, 0) = yGrid.at<uchar>(ygridID+1, xgridID+1);

	xVq.at<uchar>(0, 0) = xGrid.at<uchar>(ygridID, xgridID);
	xVq.at<uchar>(1, 0) = xGrid.at<uchar>(ygridID, xgridID + 1);
	xVq.at<uchar>(2, 0) = xGrid.at<uchar>(ygridID + 1, xgridID);
	xVq.at<uchar>(3, 0) = xGrid.at<uchar>(ygridID + 1, xgridID + 1);
}



void getLinTrans(int pstart_y, int pstart_x, Mat& yVq, Mat& xVq,Mat& T){
	// V is a 8*1 vector, p is 2*1  T is 2*8
	//
	Mat V(8, 1, CV_32FC1);
	V.at<float>(0, 0) = yVq.at<int>(0, 0); V.at<float>(1, 0) = xVq.at<int>(0, 0);
	V.at<float>(2, 0) = yVq.at<int>(1, 0); V.at<float>(3, 0) = xVq.at<int>(1, 0);
	V.at<float>(4, 0) = yVq.at<int>(2, 0); V.at<float>(5, 0) = xVq.at<int>(2, 0);
	V.at<float>(6, 0) = yVq.at<int>(3, 0); V.at<float>(7, 0) = xVq.at<int>(3, 0);

	Mat v1(2, 1, CV_32FC1), v2(2, 1, CV_32FC1), v3(2, 1, CV_32FC1), v4(2, 1, CV_32FC1);
	v1.at<float>(0, 0) = yVq.at<int>(0, 0); v1.at<float>(1, 0) = xVq.at<int>(0, 0);
	v2.at<float>(0, 0) = yVq.at<int>(1, 0); v1.at<float>(1, 0) = xVq.at<int>(1, 0);
	v3.at<float>(0, 0) = yVq.at<int>(2, 0); v1.at<float>(1, 0) = xVq.at<int>(2, 0);
	v4.at<float>(0, 0) = yVq.at<int>(3, 0); v1.at<float>(1, 0) = xVq.at<int>(3, 0);

	Mat v21 = v2 - v1,v31=v3-v1,v41=v4-v1;

	Mat p(2, 1, CV_32FC1);
	p.at<float>(0, 0) = pstart_y;  p.at<float>(1, 0) = pstart_x;
	Mat p1 = p - v1;

	double a1 = v31.at<float>(0, 0),a2=v21.at<float>(0,0),          //y
		a3 = v41.at<float>(0, 0) - v31.at<float>(0, 0) - v21.at<float>(0, 0);
	double b1 = v31.at<float>(1, 0), b2 = v21.at<float>(1, 0),      //x
		b3 = v41.at<float>(1, 0) - v31.at<float>(1, 0) - v21.at<float>(1, 0);
	double px = p1.at<float>(1, 0), py = p1.at<float>(0, 0);

	Mat tvec,mat_t;
	double t1n, t2n;
	double a, b,c;
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

	Mat out(2, 8, CV_32FC1);
	out.at<float>(0, 0) = v1w;  out.at<float>(1, 0) = 0;
	out.at<float>(0, 1) = 0;    out.at<float>(1, 1) = v1w;
	out.at<float>(0, 2) = v2w;  out.at<float>(1, 2) = 0;
	out.at<float>(0, 3) = 0;    out.at<float>(1, 3) = v2w;
	out.at<float>(0, 4) = v3w;  out.at<float>(1, 4) = 0;
	out.at<float>(0, 5) = 0;    out.at<float>(1, 5) = v3w;
	out.at<float>(0, 6) = v4w;  out.at<float>(1, 6) = 0;
	out.at<float>(0, 7) = 0;    out.at<float>(1, 7) = v4w;
	T = out;
	assert(norm(T*V - p) < 0.0001);

}





/*
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
	
	intToFloat(lineA);
	intToFloat(lineB);

	Mat A1 = lineA.rowRange(1, 2).clone();
	Mat A2 = lineA.rowRange(0, 1).clone();
	Mat deltaA 	= A1 - A2;
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

	int t1=0, t0=0;
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
	if (t1==2&&t0==2){
		isInterSect = true;
	}
	p = (lineA.rowRange(0,1) + t.at<float>(0,0)*deltaA);
}
/*
void getchannel(Mat& input_img,int n_channel,int N,Mat& output_img){
	int rows = input_img.rows;
	int cols = input_img.cols;
	Mat out(rows, cols,CV_16UC1);
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

void combin2Channel(Mat& inputImg1,Mat& inputImg2,Mat& outputImg ){
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

