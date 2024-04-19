#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void SpaceProcess(Mat& image, Mat& out_img);
Mat Mirror(Mat &image, int mode);
Mat Translate(Mat &image, int X_offset, int Y_offset, int border_mode = 1); //border_modeĬ������Ϊ1������ɫ��Ե�� ����Ϊ2������ɫ��Ե
Mat CutImage(Mat &image, int start_x, int start_y, int width, int height);
void BilinearInterpolate(Mat &image, Mat &out_image, int i, int j, double image_u, double image_v);	//��Ե���˫���Բ�ֵ
void NearestInterpolate(Mat &image, Mat &out_image, int i, int j, double image_u, double image_v);
Mat Rotate(Mat &image, float angle, int interp_mode = 1);  //interp_mode����Ϊ1��������ڲ�ֵ�� ����Ϊ2����˫���Բ�ֵ
Mat NearestInterpolate(Mat &image, int rows, int cols); //����ڲ�ֵ
Mat BilinearInterpolate(Mat &image, int rows, int cols); //˫���Բ�ֵ
