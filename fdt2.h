#pragma once
#include<opencv2/opencv.hpp>
#include <complex>
#include<windows.h>

using namespace std;
using namespace cv;

// DCT测试函数,naive表示是否采用朴素算法
void testDCT(string file, bool naive = false);

// DFT测试函数,naive表示是否采用朴素算法
void testDFT(string file, bool naive = false);

// FFT相关

// 快速傅里叶变换
void myfft(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 快速傅里叶逆变换
void myifft(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 朴素二维离散傅里叶变换
void dftNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1, bool shift = false);

// 朴素二维离散傅里叶逆变换
void idftNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 递归形式的基2一维傅里叶变换和逆变换（unscaled）
void fft1dRecurRadix2(complex<float>* src, unsigned L, bool inverse = false);

// 迭代形式的基2一维傅里叶变换和逆变换（unscaled）
void fft1dIterRadix2(complex<float>* src, unsigned L, bool inverse = false);

// CooleyTukey算法计算当L是2,3,5的倍数时的一维傅里叶变换和逆变换（unscaled）
void fft1dCooleyTukey(complex<float>* src, unsigned L, bool inverse);

// Bluestein算法计算一维傅里叶变换和逆变换（unscaled）
void fft1dBluestein(complex<float>* src, complex<float>* table, unsigned L, bool inverse);

// 朴素一维傅里叶变换和逆变换（unscaled）
void dft1dNaive(complex<float>* src, unsigned L, bool inverse);

// 一维傅里叶变换和逆变换（unscaled）
void dft1d(complex<float>* src, unsigned L, bool inverse = false);

// 利用可分离性原地计算离散傅里叶变换和逆变换（unscaled）
void helpFFT(complex<float>** src_complex, int M, int N, bool inverse = false);

// 使用FFT实现卷积
complex<float>* convolve(complex<float>* src1, complex<float>* src2, unsigned L);



// DCT相关

// 朴素二维离散余弦逆变换
void idctNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 朴素二维离散余弦变换
void dctNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 快速二维离散余弦变换
void myfdct(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 快速二维离散余弦逆变换
void myifdct(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// 朴素一维离散余弦变换
void dct1dNaive(float* src, float* dst, int N);

// 朴素一维离散余弦逆变换
void idct1dNaive(float* src, float* dst, int N);

// 基于FFT的一维DCT快速算法
void fdctfft1d(float* src, float* dst, int N);

// 基于FFT的一维IDCT快速算法
void ifdctfft1d(float* src, float* dst, int N);

// 一维离散余弦逆变换
void dct1d(float* src, float* dst, int N);

// 一维离散余弦逆变换
void idct1d(float* src, float* dst, int N);



// 工具函数

// 获取图像能够显示的幅度谱
Mat getAmplitude(Mat& src, bool logOn = false);
// 获取图像能够显示的幅度谱
Mat getPhase(Mat& src);

// 频谱中心移动到图像中央
void myfftshift(Mat& src);

// 对输入Mat做pad操作，不足补零，多余剪掉
void pad(Mat& src, int sx = -1, int sy = -1);