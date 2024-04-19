#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat BoxFilter(Mat &image, int kernel_size, bool normalize = true);	//方框滤波 normalize默认true，故为均值滤波
Mat MedianFilter(Mat &image, int kernel_size);
Mat GaussKernel(int kernel_size, double sigma); //创建高斯核
Mat GaussFilter(Mat &image, Mat &gauss_kernel);
Mat Roberts(Mat &image);
Mat Prewitt_Sobel_Laplacian(Mat &image, int mode = 1);	//mode=1 Prewitt算子 mode=2 Sobel算子 mode Laplacian算子