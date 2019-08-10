#include "pch.h"
#include <windows.h>
#include <cstdlib>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

Mat Image;
Mat Result;
Mat GrayImage;
Mat NoiseImage;
Mat SaltNoiseImage;
Mat PepperNoiseImage;
Mat GaussianNoiseImage;

void ToGrayImage()  //转换为灰度图
{
	//创建与原图同类型和同大小的矩阵
	GrayImage.create(Image.size(), Image.type());
	//将原图转换为灰度图像
	cvtColor(Image, GrayImage, CV_BGR2GRAY);
	//imshow("灰度图", GrayImage);
	waitKey();
}
void AddSaltNoise(const Mat Image, int n)  //添加盐噪声
{
	NoiseImage = Image.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % NoiseImage.rows;
		int j = rand() % NoiseImage.cols;
		//图像通道判定
		if (NoiseImage.channels() == 1)
		{
			NoiseImage.at<uchar>(i, j) = 255;		//盐噪声随机白点
		}
		else
		{
			NoiseImage.at<Vec3b>(i, j)[0] = 255;
			NoiseImage.at<Vec3b>(i, j)[1] = 255;
			NoiseImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
}
void AddPepperNoise(const Mat Image, int n)  //添加胡椒噪声
{
	NoiseImage = Image.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % NoiseImage.rows;
		int j = rand() % NoiseImage.cols;
		//图像通道判定
		if (NoiseImage.channels() == 1)
		{
			NoiseImage.at<uchar>(i, j) = 0;		//椒噪声随机黑点
		}
		else
		{
			NoiseImage.at<Vec3b>(i, j)[0] = 0;
			NoiseImage.at<Vec3b>(i, j)[1] = 0;
			NoiseImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
}
//生成高斯噪声
double generateGaussianNoise(double mu, double sigma)
{	//高斯噪声是指它的概率密度函数服从高斯分布（即正态分布）的一类噪声
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * sigma + mu;
}

void AddGaussianNoise(Mat &srcImag)  //为图像添加高斯噪声
{
	NoiseImage = srcImag.clone();
	int channels = NoiseImage.channels();
	int rowsNumber = NoiseImage.rows;
	int colsNumber = NoiseImage.cols*channels;
	//判断图像的连续性
	if (NoiseImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//添加高斯噪声
			int val = NoiseImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			NoiseImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
}
//算术均值滤波器——模板大小5*5
void ArithMeanFilter(Mat &src)  //在某时刻对信号进行连续多次采样，对采样值进行算术平均，作为该时刻的信号值
{//可以去除均匀噪声和高斯噪声，但会对图像造成一定程度的模糊。
	blur(src, Result, Size(5, 5), Point(-1, -1));
	imshow("算术均值滤波", Result);
	waitKey();
}
//几何均值滤波器——模板大小5*5
void GeoMeanFilter(Mat &src)
{//何均值滤波器能够更好的取出高斯噪声，并且能够更多的保留图像的边缘信息
	Result = src.clone();
	int row, col;
	double mul[3];
	int mn;
	if (src.channels() == 1) {
		//计算每个像素的去噪后color值
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				mul[0] = 1.0;
				mn = 0;
				//统计邻域内的几何平均值，邻域大小5*5
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						if (row >= 0 && row < src.rows && col >= 0 && col < src.cols) {
							mul[0] *= (src.at<uchar>(row, col) == 0 ? 1 : src.at<uchar>(row, col));   //邻域内的非零像素点相乘
							mn++;
						}
					}
				}
				//计算1/mn次方几何均值
				//统计成功赋给去噪后图像。
				Result.at<uchar>(i, j) = pow(mul[0], 1.0 / mn);
			}
		}
	}
	if (src.channels() == 3) {
		//计算每个像素的去噪后color值
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				mul[0] = 1.0;
				mul[1] = 1.0;
				mul[2] = 1.0;
				mn = 0;
				//统计邻域内的几何平均值，邻域大小5*5
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						if (row >= 0 && row < src.rows && col >= 0 && col < src.cols) {
							mul[0] *= (src.at<Vec3b>(row, col)[0] == 0 ? 1 : src.at<Vec3b>(row, col)[0]);   //邻域内的非零像素点相乘
							mul[1] *= (src.at<Vec3b>(row, col)[1] == 0 ? 1 : src.at<Vec3b>(row, col)[1]);   //邻域内的非零像素点相乘
							mul[2] *= (src.at<Vec3b>(row, col)[2] == 0 ? 1 : src.at<Vec3b>(row, col)[2]);   //邻域内的非零像素点相乘
							mn++;
						}
					}
				}
				//计算1/mn次方几何均值
				//统计成功赋给去噪后图像。
				Result.at<Vec3b>(i, j)[0] = pow(mul[0], 1.0 / mn);
				Result.at<Vec3b>(i, j)[1] = pow(mul[1], 1.0 / mn);
				Result.at<Vec3b>(i, j)[2] = pow(mul[2], 1.0 / mn);
			}
		}
	}
	imshow("几何均值滤波", Result);
	waitKey();
}
//谐波均值滤波器——模板大小5*5
void HarmMeanFilter(Mat &src)
{//谐波均值滤波器对盐粒噪声（白噪声）效果较好，不适用于胡椒噪声；比较适合处理高斯噪声。
	Result = src.clone();
	int row, col;
	double sum;
	int mn;
	//计算每个像素的去噪后color值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum = 0.0;
			mn = 0;
			//统计邻域,5*5模板
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row < src.rows && col >= 0 && col < src.cols) {
						sum += (src.at<uchar>(row, col) == 0 ? 255 : 255 / src.at<uchar>(row, col));
						mn++;
					}
				}
			}
			//统计成功赋给去噪后图像。
			Result.at<uchar>(i, j) = mn * 255 / sum;
		}
	}
	imshow("谐波均值滤波", Result);
	waitKey();
}
//逆谐波均值大小滤波器——模板大小5*5
void ReHarmMeanFilter(Mat &src)
{
	//以用来消除椒盐噪声。但是需要不同同时处理盐粒噪声和胡椒噪声，
	//当Q为正时，可以消除胡椒噪声；当Q为负时，消除盐粒噪声。
	//当Q=0时，该滤波器退化为算术均值滤波器；Q=-1时，退化为谐波均值滤波器。
	Result = src.clone();
	int row, col;
	double sum;
	double sum1;
	double Q = 2;
	//计算每个像素的去噪后color值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum = 0.0;
			sum1 = 0.0;
			//统计邻域
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row < src.rows && col >= 0 && col < src.cols) {
						sum += pow(src.at<uchar>(row, col) / 255, Q + 1);
						sum1 += pow(src.at<uchar>(row, col) / 255, Q);
					}
				}
			}
			//计算1/mn次方
			//统计成功赋给去噪后图像。
			Result.at<uchar>(i, j) = (sum1 == 0 ? 0 : (sum / sum1)) * 255;
		}
	}
	imshow("逆谐波均值滤波", Result);
	waitKey();
}
//中值滤波
void MFilter(Mat &src, int window) {
	medianBlur(src, Result, window);
	imshow("中值滤波", Result);
	waitKey();
}
//自适应均值滤波
//动态地改变中值滤波器的窗口尺寸，以同时兼顾去噪声作用和保护细节的效果。 
void AdaptMeanFilter(Mat &src) {
	Result = src.clone();
	blur(src, Result, Size(7, 7), Point(-1, -1));
	int row, col;
	int mn;
	double Zxy;
	double Zmed;
	double Sxy;
	double Sl;
	double Sn = 100;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Zxy = src.at<uchar>(i, j);
			Zmed = Result.at<uchar>(i, j);
			Sl = 0;
			mn = 0;
			for (int m = -3; m <= 3; m++) {
				row = i + m;
				for (int n = -3; n <= 3; n++) {
					col = j + n;
					if (row >= 0 && row < src.rows && col >= 0 && col < src.cols) {
						Sxy = src.at<uchar>(row, col);
						Sl = Sl + pow(Sxy - Zmed, 2);
						mn++;
					}
				}
			}
			Sl = Sl / mn;
			Result.at<uchar>(i, j) = Zxy - Sn / Sl * (Zxy - Zmed);
		}
	}
	imshow("自适应均值滤波", Result);
	waitKey();
}
//自适应中值滤波
void AdaptMedinFilter(Mat &src) {
	Result = src.clone();
	int row, col;
	double Zmin, Zmax, Zmed, Zxy, Smax = 7;
	int wsize;
	//计算每个像素的去噪后color值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//统计邻域
			wsize = 1;
			while (wsize <= 3) {
				Zmin = 255.0;
				Zmax = 0.0;
				Zmed = 0.0;
				Zxy = src.at<uchar>(i, j);
				int mn = 0;
				for (int m = -wsize; m <= wsize; m++) {
					row = i + m;
					for (int n = -wsize; n <= wsize; n++) {
						col = j + n;
						if (row >= 0 && row < src.rows && col >= 0 && col < src.cols) {
							double s = src.at<uchar>(row, col);
							if (s > Zmax) {
								Zmax = s;
							}
							if (s < Zmin) {
								Zmin = s;
							}
							Zmed = Zmed + s;
							mn++;
						}
					}
				}
				Zmed = Zmed / mn;
				if ((Zmed - Zmin) > 0 && (Zmed - Zmax) < 0) {
					if ((Zxy - Zmin) > 0 && (Zxy - Zmax) < 0) {
						Result.at<uchar>(i, j) = Zxy;
					}
					else {
						Result.at<uchar>(i, j) = Zmed;
					}
					break;
				}
				else {
					wsize++;
					if (wsize > 3) {
						Result.at<uchar>(i, j) = Zmed;
						break;
					}
				}
			}
		}
	}
	imshow("自适应中值滤波", Result);
	waitKey();
}
int main()
{
	

	Image = imread("lena.jpg");
	if (!Image.data)
	{
		cout << "读入图像有误！" << endl;
		system("pause");
		return -1;
	}
	ToGrayImage();	//转变为灰度图
	while (1) {
		printf("0.退出\n");
		printf("11.灰度图添加盐噪声\n");
		printf("12.灰度图添加胡椒噪声\n");
		printf("13.灰度图添加椒盐噪声\n");
		printf("14.灰度图添加高斯噪声\n");
		printf("15.彩色图添加盐噪声\n");
		printf("选择添加噪声类型：\n");
		int noise;
		cin >> noise;
		if (noise == 11) {
			AddSaltNoise(GrayImage, 300);  //灰度图添加盐噪声
		}
		else if (noise == 12) {
			AddPepperNoise(GrayImage, 300);  //灰度图添加胡椒噪声
		}
		else if (noise == 13) {
			AddSaltNoise(GrayImage, 300);
			AddPepperNoise(NoiseImage, 300);
		}
		else if (noise == 14) {
			AddGaussianNoise(GrayImage);  //灰度图添加高斯噪声
		}
		else if (noise == 15) {
			AddSaltNoise(Image, 300);  ////彩图添加盐噪声
		}
		else if (noise == 0) {
			break;
		}
		else continue;
		if (noise == 11 || noise == 12 || noise == 13 || noise == 14)
		{
			imshow("灰度原图", GrayImage);
		}
		if (noise == 15)
		{
			imshow("彩色图", Image);
		}
		imshow("有噪声图像", NoiseImage);
		waitKey();
		while (1) {
			cout << "21.算术均值滤波5*5" << endl;
			cout << "22.几何均值滤波5*5" << endl;
			cout << "23.谐波均值滤波5*5" << endl;
			cout << "24.逆谐波均值滤波5*5" << endl;
			cout << "25.中值滤波5*5" << endl;
			cout << "26.中值滤波9*9" << endl;
			cout << "27.自适应均值滤波7*7" << endl;
			cout << "28.自适应中值滤波7*7" << endl;
			cout << "29.重新添加噪声" << endl;
			cout << "选择滤波操作类型：" << endl;
			int action;
			cin >> action;
			if (action == 21) {
				ArithMeanFilter(NoiseImage);
			}
			else if (action == 22) {
				GeoMeanFilter(NoiseImage);
			}
			else if (action == 23) {
				HarmMeanFilter(NoiseImage);
			}
			else if (action == 24) {
				ReHarmMeanFilter(NoiseImage);
			}
			else if (action == 25) {
				MFilter(NoiseImage, 5);
			}
			else if (action == 26) {
				MFilter(NoiseImage, 9);
			}
			else if (action == 27) {
				AdaptMeanFilter(NoiseImage);
			}
			else if (action == 28) {
				AdaptMedinFilter(NoiseImage);
			}
			else if (action == 29) break;
			else continue;
		}
	}
	return 0;
}

