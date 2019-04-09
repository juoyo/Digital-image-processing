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

void ShowImg(Mat img, Mat grayimg);
Mat Binaryzation(Mat grayimg);
Mat Logarithm(Mat grayimg, float c);
Mat GammaTrans(Mat grayimg, float c);
Mat ColorTrans(Mat img);

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
	
	printf("可选项;\n0：退出\n11：打开灰度图\n12：灰度图二值化处理\n13：灰度图对数变换\n14：灰度图伽马变化\n15：彩色图补色变换\n");
	int method;
	while (true)
	{
		printf("选择处理方法：");
		scanf_s("%d", &method);
		//namedWindow("灰度原图", 1);
		//imshow("灰度原图", img);
		if (method == 0)
		{
			exit(-1);
		}
		else if (method == 11)
		{
			ShowImg(img, grayimg);
			destroyAllWindows();
		}
		else if (method == 12)
		{
			Mat Bresult;
			Bresult = grayimg.clone();

			Bresult = Binaryzation(grayimg);
			namedWindow("二值化图像", 2);
			imshow("二值化图像", Bresult);
			waitKey(3000);
			destroyAllWindows();
		}
		else if (method == 13)
		{
			Mat Lresult;
			Lresult = grayimg.clone();
			float c;
			printf("输入参数：");
			scanf_s("%f", &c);

			Lresult = Logarithm(grayimg, c);
			
			namedWindow("灰度图对数变换");
			imshow("灰度图对数变换", Lresult);
			waitKey(3000);
			destroyAllWindows();
		}
		else if (method == 14)
		{
			Mat Gresult;
			Gresult = grayimg.clone();
			float c;
			printf("输入参数：");
			scanf_s("%f", &c);

			Gresult = GammaTrans(grayimg, c);

			namedWindow("灰度图伽马变换");
			imshow("灰度图伽马变换", Gresult);
			waitKey(3000);
			destroyAllWindows();
		}
		else if (method == 15)
		{
			Mat Cresult;
			Cresult = img.clone();

			Cresult = ColorTrans(img);

			namedWindow("彩色图补色变换");
			imshow("彩色图补色变换", Cresult);
			waitKey(3000);
			destroyAllWindows();
		}
		else
		{
			printf("输入可选项有误，请重新输入！\n");
		}
	}
	return 0;
}

//显示原图
void ShowImg(Mat img, Mat grayimg)
{
	namedWindow("彩色原图", 1);
	imshow("彩色原图", img);
	waitKey(1500);
	namedWindow("灰度原图");
	imshow("灰度原图", grayimg);
	waitKey(1500);
}
//二值化处理
Mat Binaryzation(Mat grayimg)
{
	Mat result;
	result = grayimg.clone();
	//进行二值化处理，选择30，200.0为阈值
	threshold(grayimg, result, 128, 255.0, CV_THRESH_BINARY);
	return result;
}

//灰度图对数变化
/*
对数变换可以拓展低灰度值而压缩高灰度级值，让图像的灰度分布更加符合人眼的视觉特征。
*/
Mat Logarithm(Mat grayimg, float c)
{
	Mat result;
	result = grayimg.clone();
	double temp = 0.0;
	for (int i = 0; i < grayimg.rows; i++)
	{
		for (int j = 0; j < grayimg.cols; j++)
		{
			temp = (double)grayimg.at<uchar>(i, j);
			temp = c * log((double)(1 + temp));
			result.at<uchar>(i, j) = saturate_cast<uchar>(temp);
		}
	}
	//进行归一化处理
	normalize(result, result, 0, 255, NORM_MINMAX);
	convertScaleAbs(result, result);
	return result;
}

//灰度图伽马变换 
/*
通常来讲，当Gamma校正的值大于1时，图像的高光部分被压缩而暗调部分被扩展；
当Gamma校正的值小于1时，相反的，图像的高光部分被扩展而暗调备份被压缩。

通常情况下，最简单的Gamma校正可以用下面的幂函数来表示：
V(out) = A*V(in)^r
总之，r<1的幂函数的作用是提高图像暗区域中的对比度，而降低亮区域的对比度；
r>1的幂函数的作用是提高图像中亮区域的对比度，降低图像中按区域的对比度。
*/
Mat GammaTrans(Mat grayimg, float c)
{
	Mat result;
	result = grayimg.clone();
	//建立查表文件LUT
	unsigned char LUT[256];
	for (int i = 0; i < 256; i++)
	{
		//Gamma变换定义
		LUT[i] = saturate_cast<uchar>(pow((float)(i / 255.0), c)*255.0f);
	}
	//输入图像为单通道时，直接进行Gamma变换
	if (grayimg.channels() == 1)
	{
		MatIterator_<uchar>iterator = result.begin<uchar>();
		MatIterator_<uchar>iteratorEnd = result.end<uchar>();
		for (; iterator != iteratorEnd; iterator++)
			*iterator = LUT[(*iterator)];
	}
	else
	{
		//输入通道为3通道时，需要对每个通道分别进行变换
		MatIterator_<Vec3b>iterator = result.begin<Vec3b>();
		MatIterator_<Vec3b>iteratorEnd = result.end<Vec3b>();
		//通过查表进行转换
		for (; iterator != iteratorEnd; iterator++)
		{
			(*iterator)[0] = LUT[((*iterator)[0])];
			(*iterator)[1] = LUT[((*iterator)[1])];
			(*iterator)[2] = LUT[((*iterator)[2])];
		}
	}
	return result;
}

//彩色图补色变换
Mat ColorTrans(Mat img)
{
	Mat result;
	result = img.clone();
	//输入通道为3通道时，需要对每个通道分别进行变换
	MatIterator_<Vec3b>iterator = result.begin<Vec3b>();
	MatIterator_<Vec3b>iteratorEnd = result.end<Vec3b>();
	//通过查表进行转换
	for (; iterator != iteratorEnd; iterator++)
	{
		(*iterator)[0] = 255 - (*iterator)[0];
		(*iterator)[1] = 255 - (*iterator)[1];
		(*iterator)[2] = 255 - (*iterator)[2];
	}
	return result;
}