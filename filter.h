#pragma once
#include"fdt.h"

typedef struct NotchPairs {
	int row=0;
	int col=0;
	int r=0;
	NotchPairs(int x, int y, int z) {
		row = x;
		col = y;
		r = z;
	}
};

// 滤波器测试函数
void testFilter(string file, string type, float D0, bool gray = true, int N = 1, float D1 = 0, vector<NotchPairs>* pairs = NULL);

// 理想低通滤波器
Mat ilpf(const Mat& src, float D0);

// 巴特沃斯低通滤波器
Mat blpf(const Mat& src, float D0, int n);

// 高斯低通滤波器
Mat glpf(const Mat& src, float D0);

// 指数低通滤波器
Mat elpf(const Mat& src, float D0, int n);

// 梯形低通滤波器
Mat tlpf(const Mat& src, float D0, float D1);

// 理想高通滤波器
Mat ihpf(const Mat& src, float D0);

// 巴特沃斯高通滤波器
Mat bhpf(const Mat& src, float D0, int n);

// 高斯高通滤波器
Mat ghpf(const Mat& src, float D0);

// 指数高通滤波器
Mat ehpf(const Mat& src, float D0, int n);

// 梯形高通滤波器
Mat thpf(const Mat& src, float D0, float D1);

// 理想带通滤波器
Mat ibef(const Mat& src, float D0, float D1);

// 理想陷波滤波器
Mat inorthf(const Mat& src, vector< NotchPairs>& pairs);