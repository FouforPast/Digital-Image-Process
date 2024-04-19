#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


// 有效图片后缀
const vector<string> imgsuffixs{ "png", "jpg", "bmp", "webp", "jpeg" };

// 可选比较方法
enum COMPARE_METHOD { CORREL, CHISQR, INTERSECT, BHATTACHARYYA };


// 根据图像直方图特征检索相似图像
void mySearch(Mat& src, String& path, COMPARE_METHOD method = CORREL);

// 计算src统计的直方图的结果，将其放到一个Mat中，同时将Mat中的数据归一化到[0,high]
void calcHistVector2(Mat& src, Mat& res, double high = 100);

// 计算src统计的直方图的结果，将其放到一个向量中
void calcHistVector2(Mat& src, vector<Mat>& res);

// 对src求和
double sumScalar(Scalar src);

// 对src开方后逐个相加
double sumSqrtMat(Mat& src);

// 计算两个直方图的相似程度
float calSimilarity(Mat& src1, Mat& src2, COMPARE_METHOD method);

/*
* 显示某一张图片的直方图
* @param histogram 直方图数据，histogram[i]表示第i个通道的直方图数据，该Mat的行数是统计的直方图的灰度级个数（一般是256），列数为1，可以参考cv::calcHist()函数
* @param title 展示窗口的标题
*/
//void drawHist(vector<Mat>& histogram, string title);