#include "space_filter.h"

Mat BoxFilter(Mat &image, int kernel_size, bool normalize)	//�����˲� normalizeĬ��true����Ϊ��ֵ�˲�
{
	Mat out_image = image.clone();
	for (int i = 0; i < image.rows - kernel_size; i++)
		for (int j = 0; j < image.cols - kernel_size; j++)
			for (int c = 0; c < image.channels(); c++)
			{
				int pix_sum = 0;
				for (int m = 0; m < kernel_size; m++)
					for (int n = 0; n < kernel_size; n++)
					{
						//cout << i + m << " " << j + n << "||";
						uchar *data = image.ptr<uchar>(i + m, j + n);
						pix_sum += data[c];
					}
				uchar *new_data = out_image.ptr<uchar>(i + kernel_size / 2, j + kernel_size / 2);
				if (normalize == false) new_data[c] = min(pix_sum, 255);
				else new_data[c] = pix_sum / (kernel_size * kernel_size);
			}
	return out_image;
}

//============��ֵ�˲�==============//
Mat MedianFilter(Mat &image, int kernel_size)
{
	Mat out_image = image.clone();
	vector<uchar> vec_pix;
	for (int i = 0; i < image.rows - kernel_size; i++)
	{
		for (int j = 0; j < image.cols - kernel_size; j++)
			for (int c = 0; c < image.channels(); c++)
			{
				vec_pix.clear();
				for (int m = 0; m < kernel_size; m++)
					for (int n = 0; n < kernel_size; n++)
					{
						uchar *data = image.ptr<uchar>(i + m, j + n);
						vec_pix.push_back(data[c]);
					}
				sort(vec_pix.begin(), vec_pix.end());
				uchar *new_data = out_image.ptr<uchar>(i + kernel_size / 2, j + kernel_size / 2);
				new_data[c] = vec_pix[vec_pix.size() / 2];
			}
	}
	return out_image;
}

//===============��˹�˲�===================//
Mat GaussKernel(int kernel_size, double sigma) //������˹��
{
	Mat mask = Mat::zeros(Size(kernel_size, kernel_size), CV_64F);
	int center = kernel_size / 2;
	double sum = 0.0;
	double x, y;
	for (int i = 0; i < kernel_size; i++)
	{
		y = pow(i - center, 2);
		for (int j = 0; j < kernel_size; j++)
		{
			x = pow(j - center, 2);
			double g = exp(-(x + y) / (2 * sigma*sigma));
			mask.at<double>(i, j) = g;
			sum += g;
		}
	}
	mask = mask / sum;	//����˹�һ��
	return mask;
}

Mat GaussFilter(Mat &image, Mat &gauss_kernel)
{
	Mat out_image = image.clone();
	for (int i = 0; i < image.rows - gauss_kernel.rows; i++)
		for (int j = 0; j < image.cols - gauss_kernel.cols; j++)
			for (int c = 0; c < image.channels(); c++)
			{
				int pix_sum = 0;
				for (int m = 0; m < gauss_kernel.rows; m++)
					for (int n = 0; n < gauss_kernel.cols; n++)
					{
						//cout << i + m << " " << j + n << "||";
						uchar *data = image.ptr<uchar>(i + m, j + n);
						int pix = 0;
						pix = gauss_kernel.at<double>(n, m) * data[c];
						pix_sum += pix;
					}
				uchar *new_data = out_image.ptr<uchar>(i + gauss_kernel.rows / 2, j + gauss_kernel.rows / 2);
				new_data[c] = min(pix_sum, 255);
			}
	return out_image;
}

//==========================��======================//
//============Roberts�˲�============//
Mat Roberts(Mat &image)
{
	Mat out_image = image.clone();
	for (int i = 0; i < image.rows - 1; i++)
		for (int j = 0; j < image.cols - 1; j++)
		{
			uchar *new_data = out_image.ptr<uchar>(i, j);
			uchar *data00 = image.ptr<uchar>(i, j);
			uchar *data11 = image.ptr<uchar>(i + 1, j + 1);
			uchar *data01 = image.ptr<uchar>(i, j + 1);
			uchar *data10 = image.ptr<uchar>(i + 1, j);
			for (int c = 0; c < image.channels(); c++)
				new_data[c] = fabs(data00[c] - data11[c]) + fabs(data01[c] - data10[c]);
		}
	return out_image;
}

//===============Prewitt Sobel Laplacian�˲�================//
Mat Prewitt_Sobel_Laplacian(Mat &image, int mode)	//mode=1 Prewitt���� mode=2 Sobel���� mode=3 Laplacian����
{
	Mat out_image = image.clone();
	Mat Matrix_X, Matrix_Y;
	if (mode == 1)
	{
		Matrix_X = (Mat_<int>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);	//X�����Ե����
		Matrix_Y = (Mat_<int>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);	//Y�����Ե����	
	}
	else if (mode == 2)
	{
		Matrix_X = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);	//X�����Ե����
		Matrix_Y = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);	//Y�����Ե����	
	}
	else if (mode == 3)
	{
		Matrix_X = (Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
		Matrix_Y = (Mat_<int>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
	}
	for (int i = 0; i < image.rows - 2; i++)
		for (int j = 0; j < image.cols - 2; j++)
			for (int c = 0; c < image.channels(); c++)
			{
				int sum_X = 0, sum_Y = 0;
				for (int m = 0; m < Matrix_X.rows; m++)
					for (int n = 0; n < Matrix_X.cols; n++)
					{
						//cout << i + m << " " << j + n << "||";
						uchar *data = image.ptr<uchar>(i + m, j + n);
						sum_X += Matrix_X.at<int>(m, n) * data[c];
						sum_Y += Matrix_Y.at<int>(m, n) * data[c];
					}
				uchar *new_data = out_image.ptr<uchar>(i + Matrix_X.rows / 2, j + Matrix_X.cols / 2);
				new_data[c] = min(fabs(sum_X) + fabs(sum_Y), 255.0);
			}
	return out_image;
}