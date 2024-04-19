#include"histogram.h"

bool endsWith(string s, string sub) 
{
	return s.rfind(sub) == (s.length() - sub.length());
}

bool checkImg(string path)
{
	for (string s : imgsuffixs) {
		if (endsWith(path, s)) {
			return true;
		}
	}
	return false;
}

Mat calcGrayHist(const Mat& image, Mat& histogram)
{
	
	//注意，Size对应的是x和y，也就是第一个元数是矩阵的列数
	int rows = image.rows;   	 //输入图像的行数
	int cols = image.cols;		 //输入图像的列数
	histogram = Mat::zeros(256, 1, CV_32F);

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int index = int(image.at<uchar>(r, c));	//获取每个点的像素值
			histogram.at<float>(index, 0) += 1;			//获取了一个像素值，在相应的位置上加1
		}
	}
	return histogram;
}


// 根据图像直方图特征检索相似图像
void mySearch(Mat& src, String& path, COMPARE_METHOD method)
{
	// 输入图像的直方图
	Mat srcHist, srcHist2;
	calcHistVector2(src, srcHist);
	if (src.type() == CV_8UC3) {
		Mat tmp;
		cvtColor(src, tmp, COLOR_BGR2GRAY);
		calcHistVector2(tmp, srcHist2);
	}
	// 读取所有文件
	vector<string> files;
	glob(path, files);

	vector<pair<float, string>> res;

	for (string file : files) {
		if (!checkImg(file)) {
			continue;
		}
		Mat h;
		Mat img = imread(file, IMREAD_UNCHANGED);
		float score;
		Mat hist;
		if (src.type() != img.type()) {
			if (src.type() == CV_8U) {
				//if
				//cout<<"warning: "
				cvtColor(img, img, COLOR_BGR2GRAY);
				calcHistVector2(img, hist);
				score = calSimilarity(hist, srcHist, method);
			}
			else if(img.type() == CV_8U) {
				calcHistVector2(img, hist);
				score = calSimilarity(hist, srcHist2, method);
			}
			else {
				calcHistVector2(img, hist);
				score = calSimilarity(hist, srcHist, method);
			}
		}
		else {
			calcHistVector2(img, hist);
			score = calSimilarity(hist, srcHist, method);
		}
		res.push_back({ score, file });
	}

	sort(res.begin(), res.end(), [](pair<float, string>& a, pair<float, string>& b) {
		return a.first < b.first;
		});
	// 这两种越大越相似
	if (method == INTERSECT || method == CORREL) {
		std::reverse(res.begin(), res.end());
	}
	// 显示图片
	cout << "************************************************************************" << endl;
	int idx = 1;
	for (auto x : res) {
		printf("排名：%2d，分数：%12.5f，图像：%20s\n", idx++, x.first, x.second.c_str());
		//cout << x.first << " " << x.second << endl;
		//Mat I = imread(x.second, IMREAD_UNCHANGED);
		//imshow(x.second, I);
	}
	cout << "************************************************************************" << endl;
	destroyAllWindows();
}


void calcHistVector2(Mat& src, Mat& res, double high)
{
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	if (src.type() == CV_8U) {
		//calcHist(&src, 1, 0, Mat(), res, 1, bins, ranges);
		calcGrayHist(src, res);
		//res.convertTo(res, CV_32F, 1.0 / (src.cols * src.rows), 0);
	}
	else if (src.type() == CV_8UC3 || src.type() == CV_8UC4) {
		vector<Mat> planes_hist(3);
		vector<Mat> planes;
		split(src, planes);
		calcGrayHist(planes[0], planes_hist[0]);
		calcGrayHist(planes[1], planes_hist[1]);
		calcGrayHist(planes[2], planes_hist[2]);
		//calcHist(&planes[0], 1, 0, Mat(), planes_hist[0], 1, bins, ranges);
		//calcHist(&planes[1], 1, 0, Mat(), planes_hist[1], 1, bins, ranges);
		//calcHist(&planes[2], 1, 0, Mat(), planes_hist[2], 1, bins, ranges);
		vconcat(planes_hist, res);
		//merge(planes_hist, res);
		//res.convertTo(res, CV_32FC3, 1.0 / (src.cols * src.rows), 0);
	}
	else {
		throw std::invalid_argument("only accept CV_8U, CV_8UC4 and CV_8UC3");
	}
	normalize(res, res, 0, high, cv::NORM_MINMAX);
	//cout << res << endl;
}

void calcHistVector2(Mat& src, vector<Mat>& res)
{
	//const int bins[1] = { 256 };
	//float hranges[2] = { 0,255 };
	//const float* ranges[1] = { hranges };
	//if (src.type() == CV_8U) {
	//	res = vector<Mat>(1);
	//	calcHist(&src, 1, 0, Mat(), res, 1, bins, ranges);
	//	//res.convertTo(res, CV_32F, 1.0 / (src.cols * src.rows), 0);
	//}
	//else if (src.type() == CV_8UC3 || src.type() == CV_8UC4) {
	//	res = vector<Mat>(3);
	//	vector<Mat> planes;
	//	split(src, planes);
	//	calcHist(&planes[0], 1, 0, Mat(), res[0], 1, bins, ranges);
	//	calcHist(&planes[1], 1, 0, Mat(), res[1], 1, bins, ranges);
	//	calcHist(&planes[2], 1, 0, Mat(), res[2], 1, bins, ranges);
	//}
	//else {
	//	throw std::invalid_argument("only accept CV_8U, CV_8UC4 and CV_8UC3");
	//}
	if (src.type() == CV_8U) {
		res = vector<Mat>(1);
		calcGrayHist(src, res[0]);
		//res.convertTo(res, CV_32F, 1.0 / (src.cols * src.rows), 0);
	}
	else if (src.type() == CV_8UC3 || src.type() == CV_8UC4) {
		res = vector<Mat>(3);
		vector<Mat> planes;
		split(src, planes);
		calcGrayHist(planes[0], res[0]);
		calcGrayHist(planes[1], res[1]);
		calcGrayHist(planes[2], res[2]);
	}
	else {
		throw std::invalid_argument("only accept CV_8U, CV_8UC4 and CV_8UC3");
	}
}

double sumScalar(Scalar src)
{
	return src[0] + src[1] + src[2] + src[3];
}

// 对src开方后逐个相加
double sumSqrtMat(Mat& src)
{
	int row = src.rows, col = src.cols;
	double result = 0;
	if (src.type() == CV_8U || src.type() == CV_8UC3) {
		uchar* d = (uchar*)src.data;
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col*src.channels(); ++j) {
				result += sqrt((int)*d);
				d++;
			}
		}
	}
	else if (src.type() == CV_32F || src.type() == CV_32FC3) {
		float* d = (float*)src.data;
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				result += sqrt(*d);
				d++;
			}
		}
	}
	return result;
}



// 计算两个直方图的相似程度
float calSimilarity(Mat& src1, Mat& src2, COMPARE_METHOD method)
{
	if (src1.channels() != src2.channels()) {
		throw std::invalid_argument("src1.channels() != src2.channels()");
	}
	double score = 0;
	double opencvS;
	if (method == CORREL) {            // 相关系数
		opencvS = compareHist(src1, src2, HISTCMP_CORREL);
		int n = src1.channels();
		double tmp1 = sumScalar(cv::mean(src1)) / n;
		double tmp2 = sumScalar(cv::mean(src2)) / n;
		Scalar m1 = n == 3 ? Scalar(tmp1, tmp1, tmp1) : Scalar(tmp1);
		Scalar m2 = n == 3 ? Scalar(tmp2, tmp2, tmp2) : Scalar(tmp2);
		double Lxy = sumScalar(cv::sum((src1 - m1).mul(src2 - m2)));
		double Lxx = sumScalar(cv::sum((src1 - m1).mul(src1 - m1)));
		double Lyy = sumScalar(cv::sum((src2 - m2).mul(src2 - m2)));
		score = Lxy / sqrt(Lxx * Lyy);

		//Scalar m1 = cv::mean(src1);
		//Scalar m2 = cv::mean(src2);
		//Scalar Lxx, Lyy, Lxy;
		//Lxy = cv::sum((src1 - m1).mul(src2 - m2));
		//Lxx = cv::sum((src1 - m1).mul(src1 - m1));
		//Lyy = cv::sum((src1 - m1).mul(src2 - m2));
		//Scalar dis = Lxy * Lxy / (Lxx * Lyy);
		//return sqrt(dis[0]);
	}
	else if (method == CHISQR) {       // 卡方系数（越小越好）
		opencvS = compareHist(src1, src2, HISTCMP_CHISQR);
		int row = src1.rows, col = src1.cols;
		float* p1 = (float*)src1.data;
		float* p2 = (float*)src2.data;
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				if (*p1 != 0 || *p2 != 0) {
					score += pow(*p1 - *p2, 2) / (*p1 + *p2);
				}
				p1++;
				p2++;
			}
		}
	}
	else if (method == INTERSECT) {    //交集（越大越相似）
		opencvS = compareHist(src1, src2, HISTCMP_INTERSECT);
		Mat tmp;
		cv::min(src1, src2, tmp);
		Scalar dis = cv::sum(tmp);
		score = sumScalar(dis);
	}
	else if (method == BHATTACHARYYA) {// 巴氏距离（越小越好）
		opencvS = compareHist(src1, src2, HISTCMP_BHATTACHARYYA);
		long N = src1.rows * src1.cols * src1.channels();
		double m1 = sumScalar(cv::mean(src1)) / src1.channels();
		double m2 = sumScalar(cv::mean(src2)) / src2.channels();
		Mat tmp = (src1 / (m1 * N)).mul(src2 / (m2 * N));
		//Mat tmp = (src1).mul(src2);
		double B = sumSqrtMat(tmp);
		if (B > 1) {
			score = 0;
		}
		else {
			score = sqrt(1 - B);
		}
	}
	else {
		throw std::invalid_argument("not implemented");
	}
	return score;
}



//int main()
//{
//	Mat src1 = imread("D:\\Users\\User\\Pictures\\ucas1.jpg", IMREAD_UNCHANGED);
//	string path = "D:\\Users\\User\\Pictures";
//	Mat histogrm(256, 1, CV_32F);
//	//mySearch(src1, path, BHATTACHARYYA);
//	vector<Mat> h;
//	calcHistVector22(src1, h);
//	drawHist(h, "hist");
//	return 0;
//}

/*
* 显示某一张图片的直方图
* @param histogram 直方图数据，histogram[i]表示第i个通道的直方图数据，该Mat的行数是统计的直方图的灰度级个数（一般是256），列数为1，可以参考cv::calcHist()函数
* @param title 展示窗口的标题
*/ 
//void drawHist(vector<Mat>& histogram, string title) {
//	// 定义参数变量
//	int dims = histogram.size();
//	int bin = histogram[0].rows;
//	if (dims == 3) {
//		Mat b_hist = histogram[0];
//		Mat g_hist = histogram[1];
//		Mat r_hist = histogram[2];
//		// 显示直方图
//		int hist_w = 512;
//		int hist_h = 400;
//		int bin_w = cvRound((double)hist_w / bin);
//		Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
//		// 将直方图数据缩放
//		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//		// 绘制直方图曲线
//		for (int i = 1; i < bin; i++) {
//			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
//				Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
//			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
//				Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
//			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
//				Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
//
//		}
//		// 显示直方图
//		imshow(title, histImage);
//	}
//	else {
//		Mat hist = histogram[0];
//		// 显示直方图
//		int hist_w = 512;
//		int hist_h = 400;
//		int bin_w = cvRound((double)hist_w / bin);
//		Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
//		// 归一化直方图数据
//		normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//		// 绘制直方图曲线
//		for (int i = 1; i < bin; i++) {
//			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
//				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
//		}
//		// 显示直方图
//		imshow(title, histImage);
//	}
//	waitKey(0);
//	destroyAllWindows();
//}