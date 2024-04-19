#pragma once
#include<opencv2/opencv.hpp>
#include <complex>
#include<windows.h>

using namespace std;
using namespace cv;

// DCT���Ժ���,naive��ʾ�Ƿ���������㷨
void testDCT(string file, bool naive = false);

// DFT���Ժ���,naive��ʾ�Ƿ���������㷨
void testDFT(string file, bool naive = false);

// FFT���

// ���ٸ���Ҷ�任
void myfft(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// ���ٸ���Ҷ��任
void myifft(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// ���ض�ά��ɢ����Ҷ�任
void dftNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1, bool shift = false);

// ���ض�ά��ɢ����Ҷ��任
void idftNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// �ݹ���ʽ�Ļ�2һά����Ҷ�任����任��unscaled��
void fft1dRecurRadix2(complex<float>* src, unsigned L, bool inverse = false);

// ������ʽ�Ļ�2һά����Ҷ�任����任��unscaled��
void fft1dIterRadix2(complex<float>* src, unsigned L, bool inverse = false);

// CooleyTukey�㷨���㵱L��2,3,5�ı���ʱ��һά����Ҷ�任����任��unscaled��
void fft1dCooleyTukey(complex<float>* src, unsigned L, bool inverse);

// Bluestein�㷨����һά����Ҷ�任����任��unscaled��
void fft1dBluestein(complex<float>* src, complex<float>* table, unsigned L, bool inverse);

// ����һά����Ҷ�任����任��unscaled��
void dft1dNaive(complex<float>* src, unsigned L, bool inverse);

// һά����Ҷ�任����任��unscaled��
void dft1d(complex<float>* src, unsigned L, bool inverse = false);

// ���ÿɷ�����ԭ�ؼ�����ɢ����Ҷ�任����任��unscaled��
void helpFFT(complex<float>** src_complex, int M, int N, bool inverse = false);

// ʹ��FFTʵ�־��
complex<float>* convolve(complex<float>* src1, complex<float>* src2, unsigned L);



// DCT���

// ���ض�ά��ɢ������任
void idctNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// ���ض�ά��ɢ���ұ任
void dctNaive(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// ���ٶ�ά��ɢ���ұ任
void myfdct(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// ���ٶ�ά��ɢ������任
void myifdct(Mat& src, Mat& dst, int sx = -1, int sy = -1);

// ����һά��ɢ���ұ任
void dct1dNaive(float* src, float* dst, int N);

// ����һά��ɢ������任
void idct1dNaive(float* src, float* dst, int N);

// ����FFT��һάDCT�����㷨
void fdctfft1d(float* src, float* dst, int N);

// ����FFT��һάIDCT�����㷨
void ifdctfft1d(float* src, float* dst, int N);

// һά��ɢ������任
void dct1d(float* src, float* dst, int N);

// һά��ɢ������任
void idct1d(float* src, float* dst, int N);



// ���ߺ���

// ��ȡͼ���ܹ���ʾ�ķ�����
Mat getAmplitude(Mat& src, bool logOn = false);
// ��ȡͼ���ܹ���ʾ�ķ�����
Mat getPhase(Mat& src);

// Ƶ�������ƶ���ͼ������
void myfftshift(Mat& src);

// ������Mat��pad���������㲹�㣬�������
void pad(Mat& src, int sx = -1, int sy = -1);