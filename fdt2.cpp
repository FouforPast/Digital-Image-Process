#include"fdt.h"

#define _USE_MATH_DEFINES
#include<math.h>

unordered_map<long, complex<float>*> mem;


void testDCT(string file, bool naive)
{
    Mat I = imread(file, cv::IMREAD_GRAYSCALE);
    if (I.empty())
        return;
    //resize(I, I, Size(240, 240));

    Mat src = I.clone();
    // 补齐到最佳尺寸
    //int m = getOptimalDFTSize(I.rows);
    //int n = getOptimalDFTSize(I.cols); // on the border add zero values
    //copyMakeBorder(I, src, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    src.convertTo(src, CV_32F, 1.0 / 255);
    Mat srcDCT1, srcDCT2;
    Mat idct1, idct2;

    // opencv函数计时
    LARGE_INTEGER t1, t2, tc;
    double time;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    dct(src, srcDCT1);
    idct(srcDCT1, idct1);
    QueryPerformanceCounter(&t2);
    time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    cout << "time of opencv's dct and idct = " << time * 1000 << "ms" << endl;

    // 自编函数计时
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    if (naive) {
        dctNaive(src, srcDCT2);
        idctNaive(srcDCT2, idct2);
        //myfdct(src, srcDCT2);
        //myifdct(srcDCT2, idct2);
    }
    else {
        myfdct(src, srcDCT2);
        myifdct(srcDCT2, idct2);
    }
    QueryPerformanceCounter(&t2);
    time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    cout << "time of my dct and idct = " << time * 1000 << "ms" << endl;


    imshow("src image", src);
    imshow("opencv DCT", srcDCT1);
    imshow("my DCT", srcDCT2);
    imshow("opencv IDCT", idct1);
    imshow("my IDCT", idct2);
    I.convertTo(I, CV_32F, 1.0 / 255);
    Mat delta = idct2 - I;
    imshow("difference", delta);
    waitKey(0);
    destroyAllWindows();
}

void testDFT(string file, bool naive)
{
    Mat I = imread(file, cv::IMREAD_GRAYSCALE);
    //resize(I, I, Size(240, 240));
    if (I.empty())
        return;
    Mat padded = I.clone();
    // 补齐到最佳尺寸
    //int m = getOptimalDFTSize(I.rows);
    //int n = getOptimalDFTSize(I.cols); // on the border add zero values
    //copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    LARGE_INTEGER t1, t2, tc;
    double time;

    // 计算opencv的DFT和IDFT并计时
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI, idftI;
    merge(planes, 2, complexI);
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);

    dft(complexI, complexI);
    idft(complexI, idftI);
    //cout << complexI << endl;
    //cout << idftI << endl;
    myfftshift(complexI);
    Mat magI = getAmplitude(complexI, true);
    Mat phaI = getPhase(complexI);

    QueryPerformanceCounter(&t2);
    time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    cout << "time of opencv's dft and idft = " << time * 1000 << "ms" << endl;

    // 计算自编函数的DFT和IDFT并计时
    Mat complexI2, idftI2;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    if (naive) {
        dftNaive(padded, complexI2);
        idftNaive(complexI2, idftI2);
    }
    else {
        myfft(padded, complexI2);
        myifft(complexI2, idftI2);
    }
    QueryPerformanceCounter(&t2);
    time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    cout << "time of my dft and idft = " << time * 1000 << "ms" << endl;


    //cout << complexI2 << endl;
    //cout << idftI2 << endl;
    myfftshift(complexI2);
    Mat magI2 = getAmplitude(complexI2, true);
    Mat phaI2 = getPhase(complexI2);

    imshow("src Image", I);
    imshow("opencv DFT spectrum magnitude", magI);
    imshow("my DFT spectrum magnitude", magI2);
    imshow("opencv DFT spectrum phase", phaI);
    imshow("my DFT spectrum phase", phaI2);
    getAmplitude(idftI).convertTo(idftI, CV_8U, 1.0 / (idftI.rows * idftI.cols), 0);
    getAmplitude(idftI2).convertTo(idftI2, CV_8U);
    imshow("opencv IDFT", idftI);
    Mat delta = idftI - I;
    imshow("my IDFT", idftI2);
    imshow("difference", delta);
    waitKey();
    destroyAllWindows();
}


// 获取图像能够显示的幅度谱
Mat getAmplitude(Mat& src, bool logOn)
{
    Mat planes[2];
    split(src, planes);                          // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);  // planes[0] = magnitude  
    Mat magI = planes[0];

    if (logOn) {
        magI += Scalar::all(1);                  // 对数尺度
        log(magI, magI);
        normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    }
    return magI;
}

// 获取图像能够显示的幅度谱
Mat getPhase(Mat& src)
{
    Mat planes[2];
    split(src, planes);                          // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    phase(planes[0], planes[1], planes[0]);      // planes[0] = magnitude  
    Mat phaI = planes[0];

    normalize(phaI, phaI, 0, 1, cv::NORM_MINMAX);// 归一化到[0,1]
    return phaI;
}


// 求出n的二进制表示的最高位的1的位（从0开始）
int log2n(int n)
{
    int m = -1;
    while (n)
    {
        m++;
        n >>= 1;
    }
    return m;
}

// 朴素一维傅里叶变换和逆变换（unscaled）
void dft1dNaive(complex<float>* src, unsigned L, bool inverse)
{
    if (L == 1) {
        return;
    }
    int fact = inverse ? 1 : -1;
    complex<float>* dst = new complex<float>[L];
    for (int i = 0; i < L; ++i) {
        complex<float> w0(cos(2 * M_PI * i / L), fact * sin(2 * M_PI * i / L));
        complex<float> w(1, 0);
        for (int j = 0; j < L; ++j) {
            dst[i] += src[j] * w;
            w *= w0;
        }
    }
    for (int i = 0; i < L; ++i) {
        src[i].imag(dst[i].imag());
        src[i].real(dst[i].real());
    }
}

// 朴素一维傅里叶变换和逆变换（unscaled）
void dft1dNaive(float* real, float* imag, unsigned L, bool inverse)
{
    if (L == 1) {
        return;
    }
    int fact = inverse ? 1 : -1;
    vector<float> real_(L, 0), imag_(L, 0);
    complex<float>* dst = new complex<float>[L];
    for (int i = 0; i < L; ++i) {
        float real0 = cos(2 * M_PI * i / L), imag0 = fact * sin(2 * M_PI * i / L);
        float real1 = 1, imag1 = 0;
        for (int j = 0; j < L; ++j) {
            real_[i] += real[j] * real1 - imag[j] * imag1;
            imag_[i] += real[j] * imag1 + imag[j] * real1;
            real1 = real1 * real0 - imag1 * imag0;
            imag1 = real1 * imag0 + imag1 * real0;
        }
    }
    for (int i = 0; i < L; ++i) {
        real[i] = real_[i];
        imag[i] = imag_[i];
    }
}

// 翻转数字的bit
size_t reverseBits(size_t val, int width) {
	size_t result = 0;
	for (int i = 0; i < width; i++, val >>= 1)
		result = (result << 1) | (val & 1U);
	return result;
}

// 使用FFT实现卷积
complex<float>* convolve(complex<float>* src1, complex<float>* src2, unsigned L)
{
    fft1dIterRadix2(src1, L, false);
    fft1dIterRadix2(src2, L, false);

    complex<float>* dst = new complex<float>[L];
    for (int i = 0; i < L; ++i) {
        dst[i] = src1[i] * src2[i];
    }

    fft1dIterRadix2(dst, L, true);
    complex<float> tmp(L, 0);
    for (int i = 0; i < L; ++i) {
        dst[i] /= tmp;
    }
    return dst;
}

// Bluestein算法计算一维傅里叶变换和逆变换（unscaled）
void fft1dBluestein(complex<float>* src, complex<float>* table, unsigned L, bool inverse)
{
    // 补齐到2的幂次
    int N = 1 << (log2n(L * 2) + 1);
    int fact = inverse ? 1 : -1;
    // 预处理中间向量
    complex<float>* a = new complex<float>[N];
    complex<float>* b = new complex<float>[N];
    for (int i = 0; i < L; ++i) {
        a[i] = src[i] * table[i];
    }
    for (int i = 1; i < L * 2 - 1; ++i) {
        if (i < L) {
            b[i].imag(-table[L - i - 1].imag());
            b[i].real(table[L - i - 1].real());
        }
        else {
            b[i].imag(-table[i + 1 - L].imag());
            b[i].real(table[i + 1 - L].real());
        }
    }
    // 卷积
    complex<float>* conv = convolve(a, b, N);

    for (int i = 0; i < L; ++i) {
        src[i] = table[i] * conv[i + L];
    }
}

// 迭代形式的基2一维傅里叶变换和逆变换（unscaled）
void fft1dIterRadix2(complex<float>* src, unsigned L, bool inverse)
{
    if (L & (L - 1)) {
        throw std::invalid_argument("invalid length, make sure L is positive and a power of 2");
    }
    int n = log2n(L);
    int fact = inverse ? 1 : -1;
    // 交换原始数据的位置
    for (int i = 0; i < L; ++i) {
        size_t y = reverseBits(i, n);
        if (y > i) {
            swap(src[i], src[y]);
        }
    }
    // 预处理Wn表
    complex<float>* wn = new complex<float>[L >> 1];
    for (int i = 0; i < (L >> 1); ++i) {
        wn[i] = complex<float>(cos(2 * M_PI * i / L), fact * sin(2 * M_PI * i / L));
    }
    // 蝶形运算
    for (int i = 2; i <= L; i <<= 1) {             // 小序列长度
        for (int j = 0; j < L; j += i) {           // 小序列起始坐标
            for (int k = 0; k < (i >> 1); ++k) {   // 计算小序列的dft
                int idx1 = j + k;
                int idx2 = idx1 + (i >> 1);
                complex<float> tmp = src[idx2] * wn[k * L / i];
                src[idx2] = src[idx1] - tmp;
                src[idx1] += tmp;
            }
        }
    }
}

// 递归形式的基2一维傅里叶变换和逆变换
void fft1dRecurRadix2(complex<float>* src, unsigned L, bool inverse) {
    if (L == 1) {
        return;
    }
    if (L & (L - 1)) {
        throw std::invalid_argument("invalid length, make sure L is positive and a power of 2");
    }
    int fact = inverse ? 1 : -1;
    unsigned newL = L >> 1;
    complex<float>* even = new complex<float>[newL];
    complex<float>* odd = new complex<float>[newL];
    for (int i = 0; i < L; i += 2) {
        even[i >> 1] = src[i];
        odd[i >> 1] = src[i + 1];
    }
    fft1dRecurRadix2(even, newL, inverse);
    fft1dRecurRadix2(odd, newL, inverse);
    complex<float> w0(cos(2 * M_PI / L), fact * sin(2 * M_PI / L));
    complex<float> w(1, 0);
    for (int i = 0; i < newL; ++i) {
        complex<float> tmp = w * odd[i];
        src[i] = even[i] + tmp;
        src[i + newL] = even[i] - tmp;
        w = w * w0;
    }
}

// CooleyTukey算法计算当L是2,3,5的倍数时的一维傅里叶变换和逆变换
void fft1dCooleyTukey(complex<float>* src, unsigned L, bool inverse) {
    if (L == 1) {
        return;
    }
    int fact = inverse ? 1 : -1;
    // 确定分组数目
    int num;
    if (L % 2 == 0) {
		num = 2;
	}
	else if (L % 3 == 0) {
		num = 3;
	}
	else {
		num = 5;
	}
    // 分组
    unsigned newL = L / num;
    complex<float>** groups = new complex<float>*[num];
    for (int i = 0; i < num; ++i) {
        groups[i] = new complex<float>[newL];
    }
    for (int i = 0; i < L; ++i) {
        groups[i % num][i / num] = src[i];
    }

    // 对每组数据进行DFT
    for (int i = 0; i < num; ++i) {
        dft1d(groups[i], newL, inverse);
    }

    // 预处理表格
    //complex<float>* table3 = new complex<float>[6];
    //for (int i = 0; i < 6; ++i) {
    //    table3[i] = complex<float>(cos(2 * M_PI / 3 * i), fact * sin(2 * M_PI / 3 * i));
    //}
    //complex<float>* table5 = new complex<float>[20];
    //for (int i = 0; i < 20; ++i) {
    //    table3[i] = complex<float>(cos(2 * M_PI / 5 * i), fact * sin(2 * M_PI / 5 * i));
    //}
    
    complex<float> w0(cos(2 * M_PI / L), fact * sin(2 * M_PI / L));
    complex<float> w(1, 0);
    for (int i = 0; i < newL; ++i) {
        if (num == 2) {
            complex<float> tmp = w * groups[1][i];
            src[i] = groups[0][i] + tmp;
            src[i + newL] = groups[0][i] - tmp;
            w = w * w0;
        }
        else if (num == 3) {
            complex<float> tmp1 = w * groups[1][i];
            complex<float> tmp2 = w * w * groups[2][i];
            src[i] = groups[0][i] + tmp1 + tmp2;
            src[i + newL] = groups[0][i] + tmp1 * complex<float>(cos(2 * M_PI / 3), fact * sin(2 * M_PI / 3)) + tmp2 * complex<float>(cos(4 * M_PI / 3), fact * sin(4 * M_PI / 3));
            src[i + newL * 2] = groups[0][i] + tmp1 * complex<float>(cos(4 * M_PI / 3), fact * sin(4 * M_PI / 3)) + tmp2 * complex<float>(cos(2 * M_PI / 3), fact * sin(2 * M_PI / 3));
            w = w * w0;
        }
        //else if (num == 5) {
        //    auto tmp = w;
        //    complex<float> tmp1 = w * groups[1][i];
        //    complex<float> tmp2 = w * w * groups[2][i];
        //    complex<float> tmp3 = w * w * groups[2][i];
        //    complex<float> tmp4 = w * w * groups[2][i];
        //    src[i] = groups[0][i] + tmp1 + tmp2;
        //    src[i + newL] = groups[0][i] + tmp1 * table3[1] + tmp2 * table3[2];
        //    src[i + newL * 2] = groups[0][i] + tmp1 * table3[2] + tmp2 * table3[4];
        //    w = w * w0;
        //}
        else {
            complex<float>* temp = new complex<float>[num];
            complex<float> tmpW(1, 0);
            for (int j = 0; j < num; ++j) {
                temp[j] = groups[j][i] * tmpW;
                tmpW *= w;
            }
            dft1dNaive(temp, num, inverse);
            for (int j = 0; j < num; ++j) {
                src[i + newL * j] = temp[j];
            }
            w = w * w0;
        }
    }
}

// 获取Bluestein算法的三角函数表格
complex<float>* getBluesteinTable(unsigned L, bool inverse)
{
    int fact = inverse ? 1 : -1;
    complex<float>* table = new complex<float>[L];
    for (int i = 0; i < L; ++i) {
        uintmax_t temp = static_cast<uintmax_t>(i) * i;
        temp %= static_cast<uintmax_t>(L) * 2;
        double angle = M_PI * temp / L * fact;
        table[i].real(cos(angle));
        table[i].imag(sin(angle));
    }
    return table;
}

void dft1d(float* real, float* imag, unsigned L, bool inverse)
{
    dft1dNaive(real, imag, L, inverse);
}

// 一维傅里叶变换和逆变换（unscaled）
void dft1d(complex<float>* src, unsigned L, bool inverse)
{
    //dft1dNaive(src, L, inverse);
    //return;
    //long tmp = inverse ? -1 * L : L;
    //if (mem.count(tmp) == 0) {
    //    mem[tmp] = getBluesteinTable(L, inverse);
    //}
    //fft1dBluestein(src, mem[tmp], L, inverse);
    //return;
    if (L == 1) {
        return;
    }
    else if ((L & (L - 1)) == 0) {                      // 2的幂次的情形调用基2傅里叶变换
        //fft1dIterRadix2(src, L, inverse);
        fft1dRecurRadix2(src, L, inverse);
    }
    else {
        if (L % 5 == 0 || L % 3 == 0 || L % 2 == 0) {   // 如果是2或者3或者5的倍数，调用CooleyTukey算法
            fft1dCooleyTukey(src, L, inverse);
        }
        else {                                          // 其他情形，调用BluesteinTable算法
            long tmp = inverse ? -1 * L : L;
            if (mem.count(tmp) == 0) {
                mem[tmp] = getBluesteinTable(L, inverse);
            }
            fft1dBluestein(src, mem[tmp], L, inverse);
        }
    }
}

// 利用可分离性原地计算离散傅里叶变换和逆变换（unscaled）
void helpFFT(complex<float>** src_complex, int M, int N, bool inverse)
{
    // 对行进行FFT
    for (int i = 0; i < M; ++i) {
        dft1d(src_complex[i], N, inverse);
    }
    // 对列进行傅里叶变换
    for (int j = 0; j < N; ++j) {
        complex<float>* col = new complex<float>[M];
        for (int i = 0; i < M; ++i) {
            col[i] = src_complex[i][j];
        }
        dft1d(col, M, inverse);
        for (int i = 0; i < M; ++i) {
            src_complex[i][j] = col[i];
        }
    }
}

// 利用可分离性原地计算离散傅里叶变换和逆变换（unscaled）
void helpFFT(float* real, float* imag, int M, int N, bool inverse)
{
    // 对行进行FFT
    for (int i = 0; i < M; ++i) {
        dft1d(real + N * i, imag + N * i, N, inverse);
    }
    // 对列进行傅里叶变换
    for (int j = 0; j < N; ++j) {
        complex<float>* col = new complex<float>[M];
        float* real_ = new float[M];
        float* imag_ = new float[M];
        for (int i = 0; i < M; ++i) {
            real_[i] = real[i * M + j];
            imag_[i] = imag[i * M + j];
        }
        dft1d(real_, imag_, M, inverse);
        for (int i = 0; i < M; ++i) {
            real[i * M + j] = real_[i];
            imag[i * M + j] = imag_[i];
        }
    }
}

// 快速傅里叶变换
void myfft(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    // 将输入图像转换为复数矩阵
    complex<float>** src_complex = new complex<float>*[M];
    if (src.type() == 0) { // CV_8U
        uchar* ptr_src = (uchar*)input.data;
        for (int i = 0; i < M; i++)
        {
            src_complex[i] = new complex<float>[N];
            for (int j = 0; j < N; j++)
            {
                src_complex[i][j] = complex<float>((int)*ptr_src, 0);
                ++ptr_src;
            }
        }
    }
    else if (src.type() == 13) { //CV_32FC2
        float* ptr_src = (float*)input.data;
        for (int i = 0; i < M; i++)
        {
            src_complex[i] = new complex<float>[N];
            for (int j = 0; j < N; j++)
            {
                src_complex[i][j] = complex<float>(*(ptr_src), *(ptr_src + 1));
                ptr_src += 2;
            }
        }
    }
    else {
        throw std::invalid_argument("only CV_8U and CV_32FC2 are accepted");
    }

    helpFFT(src_complex, M, N, false);

    // 将复数矩阵转换为输出图像
    dst = Mat(M, N, CV_32FC2);
    float* ptr_dst = (float*)dst.data;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            *(ptr_dst++) = real(src_complex[i][j]);
            *(ptr_dst++) = imag(src_complex[i][j]);
        }
    }
}

// 快速傅里叶逆变换
void myifft(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    // 将输入图像转换为复数矩阵
    complex<float>** src_complex = new complex<float>*[M];
    if (src.type() == 13) { //CV_32FC2
        float* ptr_src = (float*)input.data;
        for (int i = 0; i < M; i++)
        {
            src_complex[i] = new complex<float>[N];
            for (int j = 0; j < N; j++)
            {
                src_complex[i][j] = complex<float>(*(ptr_src), *(ptr_src + 1));
                ptr_src += 2;
            }
        }
    }
    else {
        throw std::invalid_argument("only CV_32FC2 is accepted");
    }

    helpFFT(src_complex, M, N, true);

    // 将复数矩阵转换为输出图像
    dst = Mat(M, N, CV_32FC2);
    float* ptr_dst = (float*)dst.data;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            *(ptr_dst++) = real(src_complex[i][j]) / (M * N);
            *(ptr_dst++) = imag(src_complex[i][j]) / (M * N);
        }
    }
}

// 对输入Mat做pad操作，不足补零，多余剪掉
void pad(Mat& src, int sx, int sy)
{
    int M = src.rows;
    int N = src.cols;
    if (sx != -1) {
        if (sx < M) {
            src = src(Rect(0, 0, N, sx));
        }
        else {
            copyMakeBorder(src, src, 0, sx - M, 0, 0, BORDER_CONSTANT, Scalar::all(0));
        }
        M = sx;
    }
    if (sy != -1) {
        if (sy < N) {
            src = src(Rect(0, 0, sy, M));
        }
        else {
            copyMakeBorder(src, src, 0, 0, 0, sy - N, BORDER_CONSTANT, Scalar::all(0));
        }
        N = sy;
    }
    return;
}

/*
朴素二维离散傅里叶变换
args:
    src: 输入图像
    dst: 输出图像
    sx: 补齐的行数
    sy: 补齐的列数
    shift: 是否进行频谱中心化
*/
void dftNaive(Mat& src, Mat& dst, int sx, int sy, bool shift)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    if (src.type() != 0) { // CV_8U
        throw std::invalid_argument("only CV_8U is accepted");
    }

    Mat res = Mat(M, N, CV_32FC2, Scalar::all(0));
    float* ptr_res = (float*)res.data;

    for (int u = 0; u < M; ++u) {
        for (int v = 0; v < N; ++v) {
            float* ptr_src = (float*)input.data;
            for (int x = 0; x < M; ++x) {
                for (int y = 0; y < N; ++y) {
                    double tmp = ((double)u * x / (double)M + (double)v * y / (double)N) * 2 * M_PI;
                    int factor = shift & (x + y) & 1 ? -1 : 1; // 是否乘以-1
                    //*(ptr_res++) = *(ptr_src++) * cos(-tmp) * factor;;
                    //*(ptr_res++) = *(ptr_src++) * cos(-tmp) * factor;
                    res.at<Vec2f>(u, v)[0] += input.at<uchar>(x, y) * cos(-tmp) * factor;
                    res.at<Vec2f>(u, v)[1] += input.at<uchar>(x, y) * sin(-tmp) * factor;
                }
            }
            
        }
    }
    res.copyTo(dst);
}

// 朴素二维离散傅里叶逆变换
void idftNaive(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    if (src.type() != 13) { // CV_32FC2
        throw std::invalid_argument("only CV_32FC2 is accepted");
    }

    Mat res = Mat(M, N, CV_32FC2, Scalar::all(0));
    float* ptr_res = (float*)res.data;
    for (int x = 0; x < M; ++x) {
        for (int y = 0; y < N; ++y) {
            float* ptr_src = (float*)input.data;
            for (int u = 0; u < M; ++u) {
                for (int v = 0; v < N; ++v) {
                    double tmp = ((double)u * x / (double)M + (double)v * y / (double)N) * 2 * M_PI;
                    complex<float> uv{ input.at<Vec2f>(u, v)[0], input.at<Vec2f>(u, v)[1] };
                    uv = uv * complex<float>(cos(tmp), sin(tmp));  
                    res.at<Vec2f>(x, y)[0] += real(uv);
                    res.at<Vec2f>(x, y)[1] += imag(uv);

                    //double tmp = ((double)u * x / (double)M + (double)v * y / (double)N) * 2 * M_PI;
                    //complex<float> uv;
                    //uv.real(*(ptr_src++));
                    //uv.imag(*(ptr_src++));
                    //uv = uv * complex<float>(cos(tmp), sin(tmp));
                    //*(ptr_res++) += real(uv);
                    //*(ptr_res++) += imag(uv);
                }
            }
            res.at<Vec2f>(x, y)[0] /= (M * N);
            res.at<Vec2f>(x, y)[1] /= (M * N);
        }
    }
    res.copyTo(dst);
}

// 计算离散余弦变换的系数
double c(int u, int N) {
    if (u == 0) {
        return 1.0 / sqrt(N);
    }
    else {
        return sqrt(2.0 / N);
    }
}

// 朴素一维离散余弦变换
void dct1dNaive(float* src, float* dst, int N) {
	for (int u = 0; u < N; ++u) {
	    dst[u] = 0.0;
	    for (int n = 0; n < N; ++n) {
	        dst[u] += src[n] * cos(M_PI * u * (2 * n + 1) / (2 * N));
	    }
	    dst[u] *= c(u, N);
	}
}

// 朴素一维离散余弦逆变换
void idct1dNaive(float* src, float* dst, int N) {
    for (int n = 0; n < N; ++n) {
        dst[n] = 0.0;
        for (int u = 0; u < N; ++u) {
            dst[n] += c(u, N) * src[u] * cos(M_PI * u * (2 * n + 1) / (2 * N));
        }
    }
}

// 一维离散余弦逆变换
void dct1d(float* src, float* dst, int N)
{
    fdctfft1d(src, dst, N);
    //dct1dNaive(src, dst, N);
}

// 一维离散余弦逆变换
void idct1d(float* src, float* dst, int N)
{
    ifdctfft1d(src, dst, N);
    //idct1dNaive(src, dst, N);
}

// 快速二维离散余弦变换
void myfdct(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    if (src.type() != 5) {    // CV_32F
        throw std::invalid_argument("only CV_32F is accepted");
    }

    Mat res = Mat(M, N, CV_32F, Scalar::all(0));
    float* ptr_res = (float*)res.data;
    // 对行进行DCT
    for (int i = 0; i < M; ++i) {
        dct1d((float*)src.ptr(i), (float*)res.ptr(i), N);
    }
    // 对列进行DCT
    for (int j = 0; j < N; ++j) {
        float* col = new float[M];
        for (int i = 0; i < M; ++i) {
            col[i] = *(ptr_res + N * i + j);
        }
        float* tmp = new float[M];
        dct1d(col, tmp, M);
        for (int i = 0; i < M; ++i) {
            *(ptr_res + N * i + j) = tmp[i];
        }
    }
    res.copyTo(dst);
}

// 快速二维离散余弦逆变换
void myifdct(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    if (src.type() != 5) {    // CV_32F
        throw std::invalid_argument("only CV_32F is accepted");
    }

    Mat res = Mat(M, N, CV_32F, Scalar::all(0));
    float* ptr_res = (float*)res.data;
    // 对行进行IDCT
    for (int i = 0; i < M; ++i) {
        idct1d((float*)src.ptr(i), (float*)res.ptr(i), N);
    }
    // 对列进行IDCT
    for (int j = 0; j < N; ++j) {
        float* col = new float[M];
        for (int i = 0; i < M; ++i) {
            col[i] = *(ptr_res + N * i + j);
        }
        float* tmp = new float[M];
        idct1d(col, tmp, M);
        for (int i = 0; i < M; ++i) {
            *(ptr_res + N * i + j) = tmp[i];
        }
    }
    res.copyTo(dst);
}

// 基于FFT的一维DCT快速算法
void fdctfft1d(float* src, float* dst, int N) {
    //int halfLen = N / 2;
    //complex<float>* help = new complex<float>[N];
    //for (int i = 0; i < halfLen; i++) {
    //    help[i] = complex<float>(src[i * 2], 0);
    //    help[N - 1 - i] = complex<float>(src[i * 2 + 1], 0);
    //}
    //if (N % 2 == 1)
    //    help[halfLen] = complex<float>(src[N - 1], 0);
    //fft1dIterRadix2(help, N, false);
    //for (size_t i = 0; i < N; i++) {
    //    double tmp = i * M_PI / (N * 2);
    //    dst[i] = (real(help[i]) * cos(tmp) + real(help[i]) * sin(tmp)) * c(i, N);
    //}

    complex<float>* help = new complex<float>[2 * N];
    for (int i = 0; i < N; i++) {
        help[i] = complex<float>(src[i], 0);
    }
    dft1d(help, N * 2, false);
    for (int i = 0; i < N; i++) {
        double tmp = i * M_PI / (2 * N);
        dst[i] = (real(help[i]) * cos(tmp) + imag(help[i]) * sin(tmp)) * c(i, N);
    }
}


// 基于FFT的一维IDCT快速算法
void ifdctfft1d(float* src, float* dst, int N) {
    //int halfLen = N / 2;
    //complex<float>* help = new complex<float>[N];
    //help[0] = complex<float>(src[0] / 2.0, 0);
    //for (int i = 1; i < N; i++) {
    //    double tmp = i * M_PI / (N * 2);
    //    help[i] = complex<float>(src[i] * cos(tmp), -src[i] * sin(tmp));
    //}

    //fft1dIterRadix2(help, N, false);
    //for (size_t i = 0; i < halfLen; i++) {
    //    dst[i << 1] = real(help[i]);
    //    dst[(i << 1) + 1] = real(help[N - 1 - i]);
    //}
    //if (N % 2 == 1)
    //    dst[N - 1] = real(help[halfLen]);

	complex<float>* help = new complex<float>[2 * N];
	for (int i = 0; i < N; i++) {
        double tmp = i * M_PI / (2 * N);
        help[i] = complex<float>(src[i] * c(i, N) * cos(tmp), -src[i] * c(i, N) * sin(tmp));
	}
	dft1d(help, N * 2, false);
	for (int i = 0; i < N; i++) {
		dst[i] = real(help[i]);
	}
}


// 朴素二维离散余弦变换
void dctNaive(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    if (src.type() != 5) {    // CV_32F
        throw std::invalid_argument("only CV_32F is accepted");
    }

    Mat res = Mat(M, N, CV_32F, Scalar::all(0));
    float* ptr_res = (float*)res.data;
    for (int u = 0; u < M; ++u) {
        for (int v = 0; v < N; ++v) {
            double c;
            if (u == 0 && v == 0) {
                c = 0.5;
            }
            else if (u == 0 || v == 0) {
                c = sqrt(0.5);
            }
            else {
                c = 1;
            }
            float* ptr_input = (float*)input.data;
            for (int x = 0; x < M; ++x) {
                for (int y = 0; y < N; ++y) {
                    double tmp = cos((2 * x + 1) * u * M_PI / (double)(2 * M)) * cos((2 * y + 1) * v * M_PI / (double)(2 * N));
                    *ptr_res += *ptr_input * tmp;
                    ++ptr_input;
                }
            }
            *ptr_res *= c * 2 / sqrt(M * N);
            ++ptr_res;
        }
    }
    res.copyTo(dst);
}

// 朴素二维离散余弦逆变换
void idctNaive(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    if (src.type() != 5) {    // CV_32F
        throw std::invalid_argument("only CV_32F is accepted");
    }

    Mat res = Mat(M, N, CV_32F, Scalar::all(0));
    for (int x = 0; x < M; ++x) {
        for (int y = 0; y < N; ++y) {
            for (int u = 0; u < M; ++u) {
                for (int v = 0; v < N; ++v) {
                    double c;
                    if (u == 0 && v == 0) {
                        c = 0.5;
                    }
                    else if (u == 0 || v == 0) {
                        c = sqrt(0.5);
                    }
                    else {
                        c = 1;
                    }
                    double tmp = cos((2 * x + 1) * u * M_PI / (double)(2 * M)) * cos((2 * y + 1) * v * M_PI / (double)(2 * N));
                    res.at<float>(x, y) += input.at<float>(u, v) * tmp * c;
                }
            }
            res.at<float>(x, y) *= 2 / sqrt(M * N);
        }
    }
    res.copyTo(dst);
}

// 频谱中心移动到图像中央
void myfftshift(Mat& src)
{
    // crop the spectrum, if it has an odd number of rows or columns
    src = src(Rect(0, 0, src.cols & -2, src.rows & -2));
    int cx = src.cols / 2;
    int cy = src.rows / 2;

    Mat q0(src, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant 
    Mat q1(src, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(src, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(src, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}