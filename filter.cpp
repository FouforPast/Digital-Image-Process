#include"filter.h"

void testFilter(string file, string type, float D0, bool gray, int N, float D1, vector<NotchPairs>* pairs)
{
	Mat img;
	if (!gray) {
		img = imread(file, cv::IMREAD_UNCHANGED);
	}
	else {
		img = imread(file, cv::IMREAD_GRAYSCALE);
	}
	
	if (img.empty())
		return;
	//resize(img, img, Size(600, 1200));
	vector<Mat> planes_src;
	vector<Mat> planes_res;

	split(img, planes_src);

	int channel = planes_src.size();
	if (channel != 1) {
		imshow("origin image", img);
	}

	for (int i = 0; i < channel; ++i) {
		Mat padded = planes_src[i].clone();                            //expand input image to optimal size
		//int m = getOptimalDFTSize(planes_src[i].rows);
		//int n = getOptimalDFTSize(planes_src[i].cols); // on the border add zero values
		//copyMakeBorder(planes_src[i], padded, 0, m - planes_src[i].rows, 0, n - planes_src[i].cols, BORDER_CONSTANT, Scalar::all(0));

		Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
		Mat complexI;
		merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
		dft(complexI, complexI);
		//myfft(complexI, complexI);
		myfftshift(complexI);
		Mat filt, idftI, filtA, filtP, originA, originP;

		if (channel == 1) {
			imshow("origin image", planes_src[i]);
			originA = getAmplitude(complexI, true);
			originP = getPhase(complexI);
			imshow("amplitude origin", originA);
			imshow("phase origin", originP);
		}

		// 构建滤波器
		if (type == "ilpf") {
			filt = ilpf(complexI, D0);
		}
		else if (type == "blpf") {
			filt = blpf(complexI, D0, N);
		}
		else if (type == "glpf") {
			filt = glpf(complexI, D0);
		}
		else if (type == "elpf") {
			filt = elpf(complexI, D0, N);
		}
		else if (type == "tlpf") {
			filt = tlpf(complexI, D0, D1);
		}
		else if (type == "ihpf") {
			filt = ihpf(complexI, D0);
		}
		else if (type == "bhpf") {
			filt = bhpf(complexI, D0, N);
		}
		else if (type == "ghpf") {
			filt = ghpf(complexI, D0);
		}
		else if (type == "ehpf") {
			filt = ehpf(complexI, D0, N);
		}
		else if (type == "thpf") {
			filt = thpf(complexI, D0, D1);
		}
		else if (type == "ibef") {
			filt = ibef(complexI, D0, D1);
		}
		else if (type == "inorthf") {
			filt = inorthf(complexI, *pairs);
		}
		else {
			throw std::invalid_argument(type + " filter is not implemented");
		}

		if (channel == 1) {
			filtA = getAmplitude(filt, true);
			Scalar tmp;
			double min1, max1;
			cv::minMaxIdx(filtA, &min1, &max1);
			filtP = getPhase(filt);
			myfftshift(filt);
			myifft(filt, idftI);
			//idft(filt, idftI);
			//getAmplitude(idftI).convertTo(idftI, CV_8U, 1.0 / (idftI.rows * idftI.cols), 0);
			padded.convertTo(padded, CV_32F);
			idftI = getAmplitude(idftI) + 0.1 * padded;
			normalize(idftI, idftI, 0, 255, cv::NORM_MINMAX);
			idftI.convertTo(idftI, CV_8U);
			//getAmplitude(idftI).convertTo(idftI, CV_8U); 
			//equalizeHist(idftI, idftI);
			imshow("image after " + type, idftI);
			imshow("amplitude after " + type, filtA);
			imshow("phase after " + type, filtP);
		}
		else {
			myfftshift(filt);
			myifft(filt, idftI);
			padded.convertTo(padded, CV_32F);
			idftI = getAmplitude(idftI) + 0.1 * padded;
			normalize(idftI, idftI, 0, 255, cv::NORM_MINMAX);
			idftI.convertTo(idftI, CV_8U);
			equalizeHist(idftI, idftI);
			planes_res.push_back(idftI);
		}
	}

	if (channel != 1) {
		Mat result;
		merge(planes_res, result);
		//result.convertTo(result, CV_32FC3);
		//normalize(result, result, 0, 1, cv::NORM_MINMAX);
		//equalizeHist(result, result);
		imshow("image after " + type, result);
	}
	waitKey(0);
	destroyAllWindows();
}

// 理想低通滤波器
Mat ilpf(const Mat& src, float D0)
{
	int M = src.rows;
	int N = src.cols;
	D0 *= D0;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			if ((i - cx) * (i - cx) + (j - cy) * (j - cy) <= D0) {
				*(ptr_res++) = *(ptr_src++);
				*(ptr_res++) = *(ptr_src++);
			}
			else {
				ptr_res += 2;
				ptr_src += 2;
			}
		}
	}
	return res;
}

// 巴特沃斯低通滤波器
Mat blpf(const Mat& src, float D0, int n) 
{
	int M = src.rows;
	int N = src.cols;
	D0 *= D0;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float duv = (i - cx) * (i - cx) + (j - cy) * (j - cy);
			float tmp = 1 / (1 + pow(duv / D0, n));
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 高斯低通滤波器
Mat glpf(const Mat& src, float D0)
{
	int M = src.rows;
	int N = src.cols;
	D0 *= D0;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float tmp = exp(-((i - cx) * (i - cx) + (j - cy) * (j - cy)) / (2 * D0));
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 指数低通滤波器
Mat elpf(const Mat& src, float D0, int n)
{
	int M = src.rows;
	int N = src.cols;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float duv = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
			float tmp = exp(-1 * pow(duv / D0, n));
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 梯形低通滤波器
Mat tlpf(const Mat& src, float D0, float D1)
{
	if (D0 >= D1) {
		throw std::invalid_argument("D1 should be greater than D0");
	}
	int M = src.rows;
	int N = src.cols;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float duv = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
			float tmp;
			if (duv < D0) {
				tmp = 1;
			}
			else if (duv <= D1) {
				tmp = (D1 - duv) / (D1 - D0);
			}
			else {
				tmp = 0;
			}
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 理想高通滤波器
Mat ihpf(const Mat& src, float D0)
{
	int M = src.rows;
	int N = src.cols;
	D0 *= D0;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			if ((i - cx) * (i - cx) + (j - cy) * (j - cy) > D0) {
				*(ptr_res++) = *(ptr_src++);
				*(ptr_res++) = *(ptr_src++);
			}
			else {
				ptr_res += 2;
				ptr_src += 2;
			}
		}
	}
	return res;
}

// 巴特沃斯高通滤波器
Mat bhpf(const Mat& src, float D0, int n)
{
	int M = src.rows;
	int N = src.cols;
	D0 *= D0;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float duv = (i - cx) * (i - cx) + (j - cy) * (j - cy);
			float tmp = 1 / (1 + pow(D0 / duv, n));
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 高斯高通滤波器
Mat ghpf(const Mat& src, float D0)
{
	int M = src.rows;
	int N = src.cols;
	D0 *= D0;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float tmp = 1 - exp(-((i - cx) * (i - cx) + (j - cy) * (j - cy)) / (2 * D0));
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 指数高通滤波器
Mat ehpf(const Mat& src, float D0, int n)
{
	int M = src.rows;
	int N = src.cols;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float duv = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
			float tmp = exp(-1 * pow(D0 / duv, n));
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 梯形高通滤波器
Mat thpf(const Mat& src, float D0, float D1)
{
	if (D0 >= D1) {
		throw std::invalid_argument("D1 should be greater than D0");
	}
	int M = src.rows;
	int N = src.cols;
	Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	float cx = M / 2.0, cy = N / 2.0;
	if (D0 <= 0) {
		return res;
	}
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float duv = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
			float tmp;
			if (duv < D0) {
				tmp = 0;
			}
			else if (duv <= D1) {
				tmp = (duv - D0) / (D1 - D0);
			}
			else {
				tmp = 1;
			}
			*(ptr_res++) = *(ptr_src++) * tmp;
			*(ptr_res++) = *(ptr_src++) * tmp;
		}
	}
	return res;
}

// 理想陷波滤波器
Mat inorthf(const Mat& src, vector<NotchPairs>& pairs)
{
	int M = src.rows;
	int N = src.cols;
	Mat res = src.clone();
	float cx = M / 2.0, cy = N / 2.0;
	float* ptr_res = (float*)res.data;
	float* ptr_src = (float*)src.data;

	for (auto pair : pairs) {
		int y0 = pair.row, x0 = pair.col, r0 = pair.r;
		for (int y = y0 - r0; y <= y0 + r0; ++y) {
			for (int x = x0 - r0; x <= x0 + r0; ++x) {
				int duv = (y - y0) * (y - y0) + (x - x0) * (x - x0);
				if (duv <= r0 * r0) {
					res.at<Vec2f>(y, x)[0] = 0;
					res.at<Vec2f>(y, x)[1] = 0;
					//*(ptr_res + y * N * 2 + x) = 0;
					//*(ptr_res + y * N * 2 + x + 1) = 0;
				}
			}
		}
	}
	return res;
}

// 理想带阻滤波器
Mat ibef(const Mat& src, float D0, float D1)
{
	//Mat filtA = glpf(src, D0);
	//Mat filtB = ghpf(src, D1);
	//return filtA + filtB;

	//Mat filtA = blpf(src, D0, 10);
	//Mat filtB = bhpf(src, D1, 10);
	//return filtA + filtB;

	Mat filtA = ilpf(src, D0);
	Mat filtB = ihpf(src, D1);
	return filtA + filtB;

	//if (D0 > D1) {
	//	throw std::invalid_argument("D1 should be greater than D0");
	//}
	//int M = src.rows;
	//int N = src.cols;
	//Mat res = Mat::zeros(Size(N, M), CV_32FC2);
	//float cx = M / 2.0, cy = N / 2.0;
	//if (D0 <= 0) {
	//	return res;
	//}
	//float* ptr_res = (float*)res.data;
	//float* ptr_src = (float*)src.data;
	//for (int i = 0; i < M; ++i) {
	//	for (int j = 0; j < N; ++j) {
	//		float duv = sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
	//		float tmp = 1;
	//		if (duv <= D1&&duv >= D0) {
	//			tmp = 0;
	//		}
	//		*(ptr_res++) = *(ptr_src++) * tmp;
	//		*(ptr_res++) = *(ptr_src++) * tmp;
	//	}
	//}
	//return res;
}