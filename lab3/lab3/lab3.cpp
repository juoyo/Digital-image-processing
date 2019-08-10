#include "pch.h"
#include <windows.h>
#include<iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include<opencv2/core/core.hpp>
using namespace std;
using namespace cv;
int GrayMeanFilter(int x, int y);
int GrayGauFilter(int x, int y);
int GrayLaplacian();
int GrayRobert();
int GraySobel();
int ColorMeanFilter(int x, int y);
int ColorGauFilter(int x, int y);
int ColorLaplacian();
int ColorRobert();
int ColorSobel();
void EnhanceFilter(Mat img, Mat &dst, double Rate, int H, int W, int MY, int MX, float *PArray, float Model);
int ControlEnLap();

int GrayMeanFilter(int x, int y)  //灰度图均值滤波
{
	Mat srcImage = imread("lena.jpg", 0);
	if (!srcImage.data)
	{
		cout << "读入图片有误！" << endl;
		return -1;
	}
	imshow("原始图像", srcImage);
	//进行滤波相关处理
	Mat dstImage(srcImage.size(), srcImage.type());
	blur(srcImage, dstImage, Size(x, y), Point(-1, -1), 1);
	//显示效果
	imshow("均值滤波效果图", dstImage);
	waitKey(3000);
	destroyAllWindows();
	return 0;
}
int GrayGauFilter(int x, int y)  //灰度图高斯滤波
{
	Mat image, res;
	image = imread("lena.jpg", 0); 
	namedWindow("原图", WINDOW_AUTOSIZE);
	imshow("原图", image);                

	GaussianBlur(image, res, Size(x, y), 1);

	namedWindow("高斯滤波", WINDOW_AUTOSIZE); 
	imshow("高斯滤波", res);              
	waitKey(3000);
	destroyAllWindows();
	return 0;
}
int GrayLaplacian()	//Laplacian模板锐化灰度图像 
{
	Mat image, res;
	image = imread("lena.jpg", 0); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);                

	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, res, image.depth(), kernel);

	namedWindow("拉普拉斯模板", WINDOW_AUTOSIZE); 
	imshow("拉普拉斯模板", res);                
	waitKey(3000); 
	destroyAllWindows();
	return 0;
}
int GrayRobert()  //Robert模板锐化灰度图像
{
	Mat image, res;
	image = imread("lena.jpg", 0); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);               
	res = image.clone();

	for (int i = 0; i < image.rows - 1; i++) {
		for (int j = 0; j < image.cols - 1; j++) {
			//根据公式计算
			int t1 = (image.at<uchar>(i, j) -
				image.at<uchar>(i + 1, j + 1))*
				(image.at<uchar>(i, j) -
					image.at<uchar>(i + 1, j + 1));
			int t2 = (image.at<uchar>(i + 1, j) -
				image.at<uchar>(i, j + 1))*
				(image.at<uchar>(i + 1, j) -
					image.at<uchar>(i, j + 1));
			//计算g（x,y）
			res.at<uchar>(i, j) = (uchar)sqrt(t1 + t2);
		}
	}

	namedWindow("灰度图Robert滤波", WINDOW_AUTOSIZE); 
	imshow("灰度图Robert滤波", res);              
	waitKey(3000); 
	destroyAllWindows();
	return 0;
}
int GraySobel()  //Sobel模板锐化灰度图像 
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat image, res;
	image = imread("lena.jpg", 0);
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);                

	Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("X方向Sobel", abs_grad_x);

	Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("Y方向Sobel", abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
	imshow("整体Sobel", res);

	waitKey(0); 
	destroyAllWindows();
	return 0;
}

int ColorMeanFilter(int x, int y)	//彩色图均值滤波
{
	Mat image, meanRes;
	image = imread("lena.jpg", 1); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);                

	blur(image, meanRes, Size(x, y));            //均值滤波

	namedWindow("均值滤波", WINDOW_AUTOSIZE); 
	imshow("均值滤波", meanRes);                
	waitKey(3000); 
	destroyAllWindows();
	return 0;
}
int ColorGauFilter(int x, int y)	//彩色图高斯滤波
{
	Mat image, res;
	image = imread("lena.jpg", 1); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);               

	GaussianBlur(image, res, Size(x, y), 1);

	namedWindow("高斯滤波", WINDOW_AUTOSIZE);
	imshow("高斯滤波", res);               
	waitKey(3000); 
	destroyAllWindows();
	return 0;
}
int ColorLaplacian()	//Laplacian、Robert、Sobel模板锐化彩色图像 
{
	Mat image, res;
	image = imread("lena.jpg", IMREAD_COLOR); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);             

	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, res, image.depth(), kernel);

	namedWindow("拉普拉斯模板", WINDOW_AUTOSIZE); 
	imshow("拉普拉斯模板", res);             
	waitKey(3000); 
	destroyAllWindows();
	return 0;
}
int ColorRobert()	//Laplacian、Robert、Sobel模板锐化彩色图像 
{
	Mat image, res;
	image = imread("lena.jpg", 1); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);             

	res = image.clone();

	CvScalar t1, t2, t3, t4, t;
	IplImage res2 = IplImage(image);

	for (int i = 0; i < res2.height - 1; i++)
	{
		for (int j = 0; j < res2.width - 1; j++)
		{
			t1 = cvGet2D(&res2, i, j);
			t2 = cvGet2D(&res2, i + 1, j + 1);
			t3 = cvGet2D(&res2, i, j + 1);
			t4 = cvGet2D(&res2, i + 1, j);


			for (int k = 0; k < 3; k++)
			{
				int t7 = (t1.val[k] - t2.val[k])*(t1.val[k] - t2.val[k]) + (t4.val[k] - t3.val[k])*(t4.val[k] - t3.val[k]);
				t.val[k] = sqrt(t7);
			}
			cvSet2D(&res2, i, j, t);
		}
	}
	res = cvarrToMat(&res2);
	namedWindow("Robert_RGB滤波", WINDOW_AUTOSIZE); 
	imshow("Robert_RGB滤波", res);              
	waitKey(3000); 
	destroyAllWindows();
	return 0;
}
int ColorSobel()	//Laplacian、Robert、Sobel模板锐化彩色图像 
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat image, res;
	image = imread("lena.jpg", 1); 
	namedWindow("原图", WINDOW_AUTOSIZE); 
	imshow("原图", image);             

	Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("X方向Sobel", abs_grad_x);

	Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("Y方向Sobel", abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
	imshow("整体方向Sobel", res);

	waitKey(0); 
	destroyAllWindows();
	return 0;
}

void EnhanceFilter(Mat img, Mat &dst, double Rate, int H, int W, int MY, int MX, float *PArray, float Model)
{//过滤后与原图像混合
	int i, j, nHeight = img.rows, nWidth = img.cols;
	vector<vector<int>> GrayMat1, GrayMat2, GrayMat3;//暂存按比例叠加图像，R,G,B三通道
	vector<int> vecRow1(nWidth, 0), vecRow2(nWidth, 0), vecRow3(nWidth, 0);
	for (i = 0; i < nHeight; i++)
	{
		GrayMat1.push_back(vecRow1);
		GrayMat2.push_back(vecRow2);
		GrayMat3.push_back(vecRow3);
	}

	//锐化图像，并与原图像按比例叠加
	for (i = MY; i < nHeight - (H - MY) + 1; i++)
	{
		for (j = MX; j < nWidth - (W - MX) + 1; j++)
		{
			float fResult1 = 0;
			float fResult2 = 0;
			float fResult3 = 0;
			for (int k = 0; k < H; k++)
			{
				for (int l = 0; l < W; l++)
				{
					//分别计算三通道加权和
					fResult1 += img.at<Vec3b>(i, j)[0] * PArray[k*W + 1];
					fResult2 += img.at<Vec3b>(i, j)[1] * PArray[k*W + 1];
					fResult3 += img.at<Vec3b>(i, j)[2] * PArray[k*W + 1];
				}
			}

			//三通道加权和分别乘以系数并限制响应范围，最后和原图像按比例混合
			fResult1 *= Model;
			if (fResult1 > 255)
				fResult1 = 255;
			if (fResult1 < -255)
				fResult1 = -255;
			GrayMat1[i][j] = Rate * img.at<Vec3b>(i, j)[0] + fResult1 + 0.5;

			fResult2 *= Model;
			if (fResult2 > 255)
				fResult2 = 255;
			if (fResult2 < -255)
				fResult2 = -255;
			GrayMat2[i][j] = Rate * img.at<Vec3b>(i, j)[1] + fResult2 + 0.5;

			fResult3 *= Model;
			if (fResult3 > 255)
				fResult3 = 255;
			if (fResult3 < -255)
				fResult3 = -255;
			GrayMat3[i][j] = Rate * img.at<Vec3b>(i, j)[2] + fResult3 + 0.5;
		}
	}
	int nMax1 = 0, nMax2 = 0, nMax3 = 0;//三通道最大灰度和值
	int nMin1 = 65535, nMin2 = 65535, nMin3 = 65535;//三通道最小灰度和值
	//分别统计三通道最大值最小值
	for (i = MY; i < nHeight - (H - MY) + 1; i++)
	{
		for (j = MX; j < nWidth - (W - MX) + 1; j++)
		{
			if (GrayMat1[i][j] > nMax1)
				nMax1 = GrayMat1[i][j];
			if (GrayMat1[i][j] < nMin1)
				nMin1 = GrayMat1[i][j];

			if (GrayMat2[i][j] > nMax2)
				nMax2 = GrayMat2[i][j];
			if (GrayMat2[i][j] < nMin2)
				nMin2 = GrayMat2[i][j];

			if (GrayMat3[i][j] > nMax3)
				nMax3 = GrayMat3[i][j];
			if (GrayMat3[i][j] < nMin3)
				nMin3 = GrayMat3[i][j];
		}
	}
	//将按比例叠加后的三通道图像取值范围重新归一化到[0,255]
	int nSpan1 = nMax1 - nMin1, nSpan2 = nMax2 - nMin2, nSpan3 = nMax3 - nMin3;
	for (i = MY; i < nHeight - (H - MY) + 1; i++)
	{
		for (j = MX; j < nWidth - (W - MX) + 1; j++)
		{
			int br, bg, bb;
			if (nSpan1 > 0)
				br = (GrayMat1[i][j] - nMin1) * 255 / nSpan1;
			else if (GrayMat1[i][j] <= 255)
				br = GrayMat1[i][j];
			else
				br = 255;
			dst.at<Vec3b>(i, j)[0] = br;

			if (nSpan2 > 0)
				bg = (GrayMat2[i][j] - nMin2) * 255 / nSpan2;
			else if (GrayMat2[i][j] <= 255)
				bg = GrayMat2[i][j];
			else
				bg = 255;
			dst.at<Vec3b>(i, j)[1] = bg;

			if (nSpan3 > 0)
				bb = (GrayMat3[i][j] - nMin3) * 255 / nSpan3;
			else if (GrayMat3[i][j] <= 255)
				bb = GrayMat3[i][j];
			else
				bb = 255;
			dst.at<Vec3b>(i, j)[2] = bb;
		}
	}
}
float Laplac[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
int ControlEnLap()
{
	Mat image = imread("lena.jpg",1);
	namedWindow("原图", WINDOW_AUTOSIZE);
	imshow("原图", image);
	Mat result = image.clone();
	EnhanceFilter(image, result, 1.8, 3, 3, 1, 1, Laplac, 1);

	namedWindow("高提升Laplac", WINDOW_AUTOSIZE);
	imshow("高提升Laplac", result);
	waitKey(3000);
	destroyAllWindows();
	return 0;
}


int main()
{
	printf("0：退出\n11：灰度图3*3均值滤波\n12：灰度图5*5均值滤波\n13：灰度图9*9均值滤波\n");
	printf("21：灰度图3*3高斯滤波\n22：灰度图5*5高斯滤波\n23：灰度图9*9高斯滤波\n");
	printf("31：Laplacian模板锐化灰度图像\n32：Robert模板锐化灰度图像\n33：Sobel模板锐化灰度图像\n");
	printf("4：高提升滤波算法增强灰度图像\n");
	printf("51：彩色图3*3均值滤波\n52：彩色图5*5均值滤波\n53：彩色图9*9均值滤波\n");
	printf("61：彩色图3*3高斯滤波\n62：彩色图5*5高斯滤波\n63：彩色图9*9高斯滤波\n");
	printf("71：Laplacian模板锐化彩色图像\n72：Robert模板锐化彩色图像\n73：Sobel模板锐化彩色图像\n");
	int action = 0;
	int flag = 1;
	while (flag)
	{
		printf("输入选择：");
		scanf_s("%d", &action);
		switch (action)
		{
			case 11:	GrayMeanFilter(3, 3);
				break;
			case 12:	GrayMeanFilter(5, 5);
				break;
			case 13:	GrayMeanFilter(9, 9);
				break;
			case 21:	GrayGauFilter(3, 3);
				break;
			case 22:	GrayGauFilter(5, 5);
				break;
			case 23:	GrayGauFilter(9, 9);
				break;
			case 31:	GrayLaplacian();
				break;
			case 32:	GrayRobert();
				break;
			case 33:	GraySobel();
				break;
			case 4:		ControlEnLap();
				break;
			case 51:	ColorMeanFilter(3, 3);
				break;
			case 52:	ColorMeanFilter(5, 5);
				break;
			case 53:	ColorMeanFilter(9, 9);
				break;
			case 61:	ColorGauFilter(3, 3);
				break;
			case 62:	ColorGauFilter(5, 5);
				break;
			case 63:	ColorGauFilter(9, 9);
				break;
			case 71:	ColorLaplacian();
				break;
			case 72:	ColorRobert();
				break;
			case 73:	ColorSobel();
				break;
			default:
				break;
		}
		if (action == 0)
		{
			flag = 0;
		}
	}
	

	return 0;
}

