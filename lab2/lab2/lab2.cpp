#include "pch.h"
#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc/types_c.h>  
using namespace std;
using namespace cv;

Mat show_histogram(Mat img);

int main(int argc, char* argv[])
{

	const char* imgename = "lena.jpg";

	Mat img, grayimg;
	img = imread(imgename);
	grayimg = imread(imgename, 0);//以灰度图读入
	if (img.empty() || grayimg.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imgename);
		return -1;
	}
	printf("可选项;\n0：退出\n1：灰度图归一化直方图\n2：灰度图直方图均衡处理\n3：彩色图直方图均衡\n");
	
	
	int gray_sum = 0;  //像素总数
	int gray[256] = { 0 };  //记录每个灰度级别下的像素个数
	double gray_rate[256] = { 0 };  //记录灰度分布密度
	gray_sum = grayimg.rows * grayimg.cols;
	//统计每个灰度下的像素个数
	for (int i = 0; i < grayimg.rows; i++)
	{
		uchar* p = grayimg.ptr<uchar>(i);
		for (int j = 0; j < grayimg.cols; j++)
		{
			int vaule = p[j];
			gray[vaule]++;
		}
	}
	//统计灰度频率
	for (int i = 0; i < 256; i++)
	{
		gray_rate[i] = ((double)gray[i] / gray_sum);
	}
	int flag = 1;
	int method;
	while (flag)
	{
		printf("选择处理方法：");
		scanf_s("%d", &method);
		if (method == 1)	//归一化直方图
		{
			Mat result1 = grayimg.clone();
			result1 = show_histogram(result1);
			imshow("灰度直方图归一化", result1);
			waitKey(5000);
			destroyAllWindows();
		}
		else if (method == 2)	//均衡化直方图
		{
			Mat result2 = grayimg.clone();
			double gray_distribution[256] = { 0 };  //记录累计密度
			int gray_equal[256] = { 0 };  //均衡化后的灰度值
			//计算累计密度
			gray_distribution[0] = gray_rate[0];
			for (int i = 1; i < 256; i++)
			{
				gray_distribution[i] = gray_distribution[i - 1] + gray_rate[i];
			}

			//重新计算均衡化后的灰度值，四舍五入。参考公式：(N-1)*T+0.5
			for (int i = 0; i < 256; i++)
			{
				gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
			}


			//直方图均衡化,更新原图每个点的像素值
			for (int i = 0; i < result2.rows; i++)
			{
				uchar* p = result2.ptr<uchar>(i);
				for (int j = 0; j < result2.cols; j++)
				{
					p[j] = gray_equal[p[j]];
				}
			}
			result2 = show_histogram(result2);
			imshow("灰度图直方图均衡", result2);
			waitKey(5000);
			destroyAllWindows();
		}
		else if (method == 3)	//彩色直方图均衡
		{
			Mat result3 = img.clone();
			if (!result3.data) {
				cout << "failed to read" << endl;
				system("pause");
				return -1;
			}
			imshow("彩色原图", result3);
			//存储彩色直方图及图像通道向量
			Mat colorHeqImage;
			vector<Mat> BGR_plane;
			//分离BGR通道
			split(result3, BGR_plane);
			//分别对BGR通道进行直方图均衡化
			for (int i = 0; i < BGR_plane.size(); i++) {
				equalizeHist(BGR_plane[i], BGR_plane[i]);
			}
			//合并通道
			merge(BGR_plane, colorHeqImage);
			colorHeqImage = show_histogram(colorHeqImage);
			imshow("彩色图像直方图均衡", colorHeqImage);
			waitKey(5000);
			destroyAllWindows();
		}
		else
		{
			flag = 0;
		}
	}	
	return 0;
}

Mat show_histogram(Mat img)
{
	int channels = 0;
	MatND dstHist;	
	int histSize[] = { 256 };      	
	float midRanges[] = { 0, 255 };
	const float *ranges[] = { midRanges };
	calcHist(&img, 1, &channels, Mat(), dstHist, 1, histSize, ranges, true, false);
	Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	double MaxValue;
	minMaxLoc(dstHist, 0, &MaxValue, 0, 0);  //图像最大值
	for (int i = 0; i < 256; i++)
	{
		int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / MaxValue);  //四舍五入
		line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 255, 255));
	}
	
	return drawImage;
}

