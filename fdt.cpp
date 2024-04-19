#include"fdt.h"

#define _USE_MATH_DEFINES
#include<math.h>

unordered_map<long, complex<float>*> mem;


void testDCT(string file, bool naive)
{
    Mat I = imread(file, cv::IMREAD_GRAYSCALE);
    if (I.empty())
        return;
    //resize(I, I, Size(256, 256));

    Mat src = I.clone();
    // ���뵽��ѳߴ�
    //int m = getOptimalDFTSize(I.rows);
    //int n = getOptimalDFTSize(I.cols); // on the border add zero values
    //copyMakeBorder(I, src, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    src.convertTo(src, CV_32F, 1.0 / 255);
    Mat srcDCT1, srcDCT2;
    Mat idct1, idct2;

    // opencv������ʱ
    LARGE_INTEGER t1, t2, tc;
    double time;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    dct(src, srcDCT1);
    idct(srcDCT1, idct1);
    QueryPerformanceCounter(&t2);
    time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
    cout << "time of opencv's dct and idct = " << time * 1000 << "ms" << endl;

    // �Աຯ����ʱ
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
    //resize(I, I, Size(257, 257));
    if (I.empty())
        return;
    Mat padded = I.clone();
    // ���뵽��ѳߴ�
    //int m = getOptimalDFTSize(I.rows);
    //int n = getOptimalDFTSize(I.cols); // on the border add zero values
    //copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    LARGE_INTEGER t1, t2, tc;
    double time;

    // ����opencv��DFT��IDFT����ʱ
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

    // �����Աຯ����DFT��IDFT����ʱ
    Mat complexI2, idftI2;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    if (naive) {
        //myfft(padded, complexI2);
        //myifft(complexI2, idftI2);
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


// ��ȡͼ���ܹ���ʾ�ķ�����
Mat getAmplitude(Mat& src, bool logOn)
{
    Mat planes[2];
    split(src, planes);                          // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);  // planes[0] = magnitude  
    Mat magI = planes[0];

    if (logOn) {
        magI += Scalar::all(1);                  // �����߶�
        log(magI, magI);
        normalize(magI, magI, 0, 1, cv::NORM_MINMAX);
    }
    return magI;
}

// ��ȡͼ���ܹ���ʾ�ķ�����
Mat getPhase(Mat& src)
{
    Mat planes[2];
    split(src, planes);                          // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    phase(planes[0], planes[1], planes[0]);      // planes[0] = magnitude  
    Mat phaI = planes[0];

    normalize(phaI, phaI, 0, 1, cv::NORM_MINMAX);// ��һ����[0,1]
    return phaI;
}


// ���n�Ķ����Ʊ�ʾ�����λ��1��λ����0��ʼ��
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


// ����һά����Ҷ�任����任��unscaled��
void dft1dNaive(float* real, float* imag, unsigned L, bool inverse)
{
    if (L == 1) {
        return;
    }
    int fact = inverse ? 1 : -1;
    vector<double> real_(L, 0), imag_(L, 0);
    for (int i = 0; i < L; ++i) {
        double real0 = cos(2 * M_PI * i / L), imag0 = fact * sin(2 * M_PI * i / L);
        double real1 = 1, imag1 = 0;
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

// ��ת���ֵ�bit
size_t reverseBits(size_t val, int width) {
	size_t result = 0;
	for (int i = 0; i < width; i++, val >>= 1)
		result = (result << 1) | (val & 1U);
	return result;
}

// ʹ��FFTʵ�־��
float** convolve(float* real1, float* imag1, float* real2, float* imag2, unsigned L)
{
    fft1dIterRadix2(real1, imag1, L, false);
    fft1dIterRadix2(real2, imag2, L, false);

    float** dst = new float*[2]();
    dst[0] = new float[L];
    dst[1] = new float[L];
    for (int i = 0; i < L; ++i) {
        dst[0][i] = real1[i] * real2[i] - imag1[i] * imag2[i];
        dst[1][i] = imag1[i] * real2[i] + real1[i] * imag2[i];
    }

    fft1dIterRadix2(dst[0], dst[1], L, true);
    for (int i = 0; i < L; ++i) {
        dst[0][i] /= L;
        dst[1][i] /= L;
    }
    return dst;
}

// Bluestein�㷨����һά����Ҷ�任����任��unscaled��
void fft1dBluestein(float* real, float* imag, complex<float>* table, unsigned L, bool inverse)
{
    // ���뵽2���ݴ�
    int N = 1 << (log2n(L * 2) + 1);
    int fact = inverse ? 1 : -1;
    // Ԥ�����м�����
    float* realA = new float[N]();
    float* realB = new float[N]();    
    float* imagA = new float[N]();
    float* imagB = new float[N]();

    //memset(realA, 0, sizeof(float) * N);
    //memset(realB, 0, sizeof(float) * N);
    //memset(imagA, 0, sizeof(float) * N);
    //memset(imagB, 0, sizeof(float) * N);

    complex<float>* a = new complex<float>[N];
    complex<float>* b = new complex<float>[N];
    for (int i = 0; i < L; ++i) {
        realA[i] = real[i] * table[i].real() - imag[i] * table[i].imag();
        imagA[i] = imag[i] * table[i].real() + real[i] * table[i].imag();

    }
    for (int i = 1; i < L * 2 - 1; ++i) {
        if (i < L) {
            imagB[i] = -table[L - i - 1].imag();
            realB[i] = table[L - i - 1].real();
        }
        else {
            imagB[i] = -table[i + 1 - L].imag();
            realB[i] = table[i + 1 - L].real();
        }
    }
    // ���
    float** conv = convolve(realA, imagA, realB, imagB, N);

    for (int i = 0; i < L; ++i) {
        real[i] = conv[0][i + L] * table[i].real() - conv[1][i + L] * table[i].imag();
        imag[i] = conv[1][i + L] * table[i].real() + conv[0][i + L] * table[i].imag();
    }
}

// ������ʽ�Ļ�2һά����Ҷ�任����任��unscaled��
void fft1dIterRadix2(float* real, float* imag, unsigned L, bool inverse)
{
    if (L & (L - 1)) {
        throw std::invalid_argument("invalid length, make sure L is positive and a power of 2");
    }
    int n = log2n(L);
    int fact = inverse ? 1 : -1;
    // ����ԭʼ���ݵ�λ��
    for (int i = 0; i < L; ++i) {
        size_t y = reverseBits(i, n);
        if (y > i) {
            swap(real[i], real[y]);
            swap(imag[i], imag[y]);
        }
    }
    // Ԥ����Wn��
    float* _real = new float[L >> 1];
    float* _imag = new float[L >> 1];

    for (int i = 0; i < (L >> 1); ++i) {
        _real[i] = cos(2 * M_PI * i / L);
        _imag[i] = fact * sin(2 * M_PI * i / L);
    }
    // ��������
    for (int i = 2; i <= L; i <<= 1) {             // С���г���
        for (int j = 0; j < L; j += i) {           // С������ʼ����
            for (int k = 0; k < (i >> 1); ++k) {   // ����С���е�dft
                int idx1 = j + k;
                int idx2 = idx1 + (i >> 1);
                float tmpReal = real[idx2] * _real[k * L / i] - imag[idx2] * _imag[k * L / i];
                float tmpImag = imag[idx2] * _real[k * L / i] + real[idx2] * _imag[k * L / i];
                real[idx2] = real[idx1] - tmpReal;
                imag[idx2] = imag[idx1] - tmpImag;
                real[idx1] += tmpReal;
                imag[idx1] += tmpImag;
            }
        }
    }
}

// �ݹ���ʽ�Ļ�2һά����Ҷ�任����任
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

// CooleyTukey�㷨���㵱L��2,3,5�ı���ʱ��һά����Ҷ�任����任
void fft1dCooleyTukey(float* real, float* imag, unsigned L, bool inverse) {
    if (L == 1) {
        return;
    }
    int fact = inverse ? 1 : -1;
    // ȷ��������Ŀ
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
    // ����
    unsigned newL = L / num;
    float** groupReal = new float* [num];
    float** groupImag = new float* [num];

    for (int i = 0; i < num; ++i) {
        groupReal[i] = new float[newL];
        groupImag[i] = new float[newL];
    }
    for (int i = 0; i < L; ++i) {
        groupReal[i % num][i / num] = real[i];
        groupImag[i % num][i / num] = imag[i];
    }

    // ��ÿ�����ݽ���DFT
    for (int i = 0; i < num; ++i) {
        dft1d(groupReal[i], groupImag[i], newL, inverse);
    }

    // Ԥ������
    //complex<float>* table3 = new complex<float>[6];
    //for (int i = 0; i < 6; ++i) {
    //    table3[i] = complex<float>(cos(2 * M_PI / 3 * i), fact * sin(2 * M_PI / 3 * i));
    //}
    //complex<float>* table5 = new complex<float>[20];
    //for (int i = 0; i < 20; ++i) {
    //    table3[i] = complex<float>(cos(2 * M_PI / 5 * i), fact * sin(2 * M_PI / 5 * i));
    //}
    
    //complex<float> w0(cos(2 * M_PI / L), fact * sin(2 * M_PI / L));
    //complex<float> w(1, 0);
    float real0 = cos(2 * M_PI / L), imag0 = fact * sin(2 * M_PI / L);
    float real1 = 1, imag1 = 0;
    for (int i = 0; i < newL; ++i) {
        if (num == 2) {
            float tmpReal = real1 * groupReal[1][i] - imag1 * groupImag[1][i];
            float tmpImag = imag1 * groupReal[1][i] + real1 * groupImag[1][i];
            real[i] = groupReal[0][i] + tmpReal;
            imag[i] = groupImag[0][i] + tmpImag;
            real[i + newL] = groupReal[0][i] - tmpReal;
            imag[i + newL] = groupImag[0][i] - tmpImag;
        }
        //else if (num == 3) {
        //    complex<float> tmp1 = w * groups[1][i];
        //    complex<float> tmp2 = w * w * groups[2][i];
        //    src[i] = groups[0][i] + tmp1 + tmp2;
        //    src[i + newL] = groups[0][i] + tmp1 * complex<float>(cos(2 * M_PI / 3), fact * sin(2 * M_PI / 3)) + tmp2 * complex<float>(cos(4 * M_PI / 3), fact * sin(4 * M_PI / 3));
        //    src[i + newL * 2] = groups[0][i] + tmp1 * complex<float>(cos(4 * M_PI / 3), fact * sin(4 * M_PI / 3)) + tmp2 * complex<float>(cos(2 * M_PI / 3), fact * sin(2 * M_PI / 3));
        //    w = w * w0;
        //}
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
            float* tempReal = new float[num];
            float* tempImag = new float[num];
            float real2 = 1, imag2 = 0;
            for (int j = 0; j < num; ++j) {
                tempReal[j] = groupReal[j][i] * real2 - groupImag[j][i] * imag2;
                tempImag[j] = groupImag[j][i] * real2 + groupReal[j][i] * imag2;
                auto tmpReal2 = real2;
                real2 = real2 * real1 - imag2 * imag1;
                imag2 = tmpReal2 * imag1 + imag2 * real1;
            }
            dft1dNaive(tempReal, tempImag, num, inverse);
            for (int j = 0; j < num; ++j) {
                real[i + newL * j] = tempReal[j];
                imag[i + newL * j] = tempImag[j];
            }
        }
        auto tmpReal1 = real1;
        real1 = real1 * real0 - imag1 * imag0;
        imag1 = tmpReal1 * imag0 + imag1 * real0;
    }
}

// ��ȡBluestein�㷨�����Ǻ������
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
    //dft1dNaive(real, imag, L, inverse);
    //return;
    if (L == 1) {
        return;
    }
    else if ((L & (L - 1)) == 0) {                      // 2���ݴε����ε��û�2����Ҷ�任
        //fft1dIterRadix2(real, imag, L, inverse);
        //fft1dCooleyTukey(real, imag, L, inverse);
        //fft1dRecurRadix2(real, imag, L, inverse);
        long tmp = inverse ? -1 * L : L;
        if (mem.count(tmp) == 0) {
            mem[tmp] = getBluesteinTable(L, inverse);
        }
        fft1dBluestein(real, imag, mem[tmp], L, inverse);
    }
    else {
        if (L % 5 == 0 || L % 3 == 0 || L % 2 == 0) {   // �����2����3����5�ı���������CooleyTukey�㷨
            fft1dCooleyTukey(real, imag, L, inverse);
        }
        else {                                          // �������Σ�����BluesteinTable�㷨
            long tmp = inverse ? -1 * L : L;
            if (mem.count(tmp) == 0) {
                mem[tmp] = getBluesteinTable(L, inverse);
            }
            fft1dBluestein(real, imag, mem[tmp], L, inverse);
        }
    }
}


// ���ÿɷ�����ԭ�ؼ�����ɢ����Ҷ�任����任��unscaled��
void helpFFT(float* real, float* imag, int M, int N, bool inverse)
{
    // ���н���FFT
    for (int i = 0; i < M; ++i) {
        dft1d(real + N * i, imag + N * i, N, inverse);
    }
    // ���н��и���Ҷ�任
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

// ���ٸ���Ҷ�任
void myfft(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    Mat real, imag;
    if (src.type() == 0) { // CV_8U
        src.convertTo(real, CV_32F);
        imag = Mat::zeros(src.size(), CV_32F);
    }
    else if (src.type() == 13) { //CV_32FC2
        Mat planes[2];
        split(src, planes);
        imag = planes[0];
        real = planes[1];
    }
    else {
        throw std::invalid_argument("only CV_8U and CV_32FC2 are accepted");
    }
    float* ptr_real = (float*)real.data;
    float* ptr_imag = (float*)imag.data;

    helpFFT(ptr_real, ptr_imag, M, N, false);

    // ����������ת��Ϊ���ͼ��
    dst = Mat(M, N, CV_32FC2);
    float* ptr_dst = (float*)dst.data;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            *(ptr_dst++) = *(ptr_real++);
            *(ptr_dst++) = *(ptr_imag++);
        }
    }
}

// ���ٸ���Ҷ��任
void myifft(Mat& src, Mat& dst, int sx, int sy)
{
    Mat input = src.clone();
    pad(input, sx, sy);
    int M = input.rows;
    int N = input.cols;

    Mat real, imag;
    if (src.type() == 13) { //CV_32FC2
        Mat planes[2];
        split(src, planes);
        imag = planes[0];
        real = planes[1];
    }
    else {
        throw std::invalid_argument("only CV_32FC2 is accepted");
    }

    float* ptr_real = (float*)real.data;
    float* ptr_imag = (float*)imag.data;

    helpFFT(ptr_real, ptr_imag, M, N, false);

    // ����������ת��Ϊ���ͼ��
    dst = Mat(M, N, CV_32FC2);
    float* ptr_dst = (float*)dst.data;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            *(ptr_dst++) = *(ptr_real++) / (M * N);
            *(ptr_dst++) = *(ptr_imag++) / (M * N);
        }
    }
}

// ������Mat��pad���������㲹�㣬�������
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

// ���ض�ά��ɢ����Ҷ�任
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
                    int factor = shift & (x + y) & 1 ? -1 : 1; // �Ƿ����-1
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

// ���ض�ά��ɢ����Ҷ��任
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

// ������ɢ���ұ任��ϵ��
double c(int u, int N) {
    if (u == 0) {
        return 1.0 / sqrt(N);
    }
    else {
        return sqrt(2.0 / N);
    }
}

// ����һά��ɢ���ұ任
void dct1dNaive(float* src, float* dst, int N) {
	for (int u = 0; u < N; ++u) {
	    dst[u] = 0.0;
	    for (int n = 0; n < N; ++n) {
	        dst[u] += src[n] * cos(M_PI * u * (2 * n + 1) / (2 * N));
	    }
	    dst[u] *= c(u, N);
	}
	//// ������ɢ���ұ任��FDCT�������η�ʵ��
	//if (N == 1) {
	//	// ����źų���Ϊ 1����ֱ�ӽ��źŸ�ֵ�����
	//	X[0] = x[0];
	//}
	//else {
	//	// ���źŷֳ����룬�ֱ������ɢ���ұ任
	//	int N2 = N / 2;
	//	float *x1 = new float[N2], *x2 = new float[N2];
	//	for (int i = 0; i < N2; ++i) {
	//		x1[i] = x[i];
	//		x2[i] = x[i + N2];
	//	}
 //       float* X1 = new float[N2], * X2 = new float[N2];
 //       fdct1d(x1, X1, N2);
 //       fdct1d(x2, X2, N2);

	//	// ����������ϲ�Ϊ���յ���ɢ���ұ任���
	//	for (int u = 0; u < N; ++u) {
	//		if (u < N2) {
 //               X[u] = (X1[u] + cos(M_PI * u / N) * X2[u]);
	//		}
	//		else {
 //               X[u] = (X1[u - N2] - sin(M_PI * u / N) * X2[u - N2]);
	//		}
 //           X[u] *= c(u, N);
	//	}
	//}
}

// ����һά��ɢ������任
void idct1dNaive(float* src, float* dst, int N) {
    for (int n = 0; n < N; ++n) {
        dst[n] = 0.0;
        for (int u = 0; u < N; ++u) {
            dst[n] += c(u, N) * src[u] * cos(M_PI * u * (2 * n + 1) / (2 * N));
        }
    }
}

// һά��ɢ������任
void dct1d(float* src, float* dst, int N)
{
    //fdctfft1d(src, dst, N);
    dct1dNaive(src, dst, N);
    //if (N & (N - 1)) {
    //    dct1dNaive(src, dst, N);
    //}
    //else {
    //    fdctfft1d(src, dst, N);
    //}
}

// һά��ɢ������任
void idct1d(float* src, float* dst, int N)
{
    //ifdctfft1d(src, dst, N);
    idct1dNaive(src, dst, N);
    //if (N & (N - 1)) {
    //    idct1dNaive(src, dst, N);
    //}
    //else {
    //    ifdctfft1d(src, dst, N);
    //}
}

// ���ٶ�ά��ɢ���ұ任
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
    // ���н���DCT
    for (int i = 0; i < M; ++i) {
        dct1d((float*)src.ptr(i), (float*)res.ptr(i), N);
    }
    // ���н���DCT
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

// ���ٶ�ά��ɢ������任
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
    // ���н���IDCT
    for (int i = 0; i < M; ++i) {
        idct1d((float*)src.ptr(i), (float*)res.ptr(i), N);
    }
    // ���н���IDCT
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

// ����FFT��һάDCT�����㷨
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
    float* imag = new float[2 * N];
    float* real = new float[2 * N];
    memset(imag, 0, N * 2);
    memset(real, 0, N * 2);
    for (int i = 0; i < N; i++) {
        real[i] = src[i];
    }
    dft1d(real, imag, N * 2, false);
    for (int i = 0; i < N; i++) {
        double tmp = i * M_PI / (2 * N);
        dst[i] = (real[i] * cos(tmp) + imag[i] * sin(tmp)) * c(i, N);
    }
}


// ����FFT��һάIDCT�����㷨
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

    float* imag = new float[2 * N];
    float* real = new float[2 * N];
    memset(imag, 0, N * 2);
    memset(real, 0, N * 2);
    for (int i = 0; i < N; i++) {
        double tmp = i * M_PI / (2 * N);
        real[i] = src[i] * c(i, N) * cos(tmp);
        imag[i] = -src[i] * c(i, N) * sin(tmp);
    }
    dft1d(real, imag, N * 2, false);
    for (int i = 0; i < N; i++) {
        dst[i] = real[i];
    }
}


// ���ض�ά��ɢ���ұ任
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

// ���ض�ά��ɢ������任
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

// Ƶ�������ƶ���ͼ������
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