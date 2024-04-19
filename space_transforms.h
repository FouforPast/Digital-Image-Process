#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void SpaceProcess(Mat& image, Mat& out_img);
Mat Mirror(Mat &image, int mode);
Mat Translate(Mat &image, int X_offset, int Y_offset, int border_mode = 1); //border_mode默认设置为1，填充黑色边缘， 设置为2，填充白色边缘
Mat CutImage(Mat &image, int start_x, int start_y, int width, int height);
void BilinearInterpolate(Mat &image, Mat &out_image, int i, int j, double image_u, double image_v);	//针对单点双线性插值
void NearestInterpolate(Mat &image, Mat &out_image, int i, int j, double image_u, double image_v);
Mat Rotate(Mat &image, float angle, int interp_mode = 1);  //interp_mode设置为1采用最近邻插值， 设置为2采用双线性插值
Mat NearestInterpolate(Mat &image, int rows, int cols); //最近邻插值
Mat BilinearInterpolate(Mat &image, int rows, int cols); //双线性插值
