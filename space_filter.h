#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat BoxFilter(Mat &image, int kernel_size, bool normalize = true);	//�����˲� normalizeĬ��true����Ϊ��ֵ�˲�
Mat MedianFilter(Mat &image, int kernel_size);
Mat GaussKernel(int kernel_size, double sigma); //������˹��
Mat GaussFilter(Mat &image, Mat &gauss_kernel);
Mat Roberts(Mat &image);
Mat Prewitt_Sobel_Laplacian(Mat &image, int mode = 1);	//mode=1 Prewitt���� mode=2 Sobel���� mode Laplacian����