#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


// ��ЧͼƬ��׺
const vector<string> imgsuffixs{ "png", "jpg", "bmp", "webp", "jpeg" };

// ��ѡ�ȽϷ���
enum COMPARE_METHOD { CORREL, CHISQR, INTERSECT, BHATTACHARYYA };


// ����ͼ��ֱ��ͼ������������ͼ��
void mySearch(Mat& src, String& path, COMPARE_METHOD method = CORREL);

// ����srcͳ�Ƶ�ֱ��ͼ�Ľ��������ŵ�һ��Mat�У�ͬʱ��Mat�е����ݹ�һ����[0,high]
void calcHistVector2(Mat& src, Mat& res, double high = 100);

// ����srcͳ�Ƶ�ֱ��ͼ�Ľ��������ŵ�һ��������
void calcHistVector2(Mat& src, vector<Mat>& res);

// ��src���
double sumScalar(Scalar src);

// ��src������������
double sumSqrtMat(Mat& src);

// ��������ֱ��ͼ�����Ƴ̶�
float calSimilarity(Mat& src1, Mat& src2, COMPARE_METHOD method);

/*
* ��ʾĳһ��ͼƬ��ֱ��ͼ
* @param histogram ֱ��ͼ���ݣ�histogram[i]��ʾ��i��ͨ����ֱ��ͼ���ݣ���Mat��������ͳ�Ƶ�ֱ��ͼ�ĻҶȼ�������һ����256��������Ϊ1�����Բο�cv::calcHist()����
* @param title չʾ���ڵı���
*/
//void drawHist(vector<Mat>& histogram, string title);