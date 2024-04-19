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

// �˲������Ժ���
void testFilter(string file, string type, float D0, bool gray = true, int N = 1, float D1 = 0, vector<NotchPairs>* pairs = NULL);

// �����ͨ�˲���
Mat ilpf(const Mat& src, float D0);

// ������˹��ͨ�˲���
Mat blpf(const Mat& src, float D0, int n);

// ��˹��ͨ�˲���
Mat glpf(const Mat& src, float D0);

// ָ����ͨ�˲���
Mat elpf(const Mat& src, float D0, int n);

// ���ε�ͨ�˲���
Mat tlpf(const Mat& src, float D0, float D1);

// �����ͨ�˲���
Mat ihpf(const Mat& src, float D0);

// ������˹��ͨ�˲���
Mat bhpf(const Mat& src, float D0, int n);

// ��˹��ͨ�˲���
Mat ghpf(const Mat& src, float D0);

// ָ����ͨ�˲���
Mat ehpf(const Mat& src, float D0, int n);

// ���θ�ͨ�˲���
Mat thpf(const Mat& src, float D0, float D1);

// �����ͨ�˲���
Mat ibef(const Mat& src, float D0, float D1);

// �����ݲ��˲���
Mat inorthf(const Mat& src, vector< NotchPairs>& pairs);