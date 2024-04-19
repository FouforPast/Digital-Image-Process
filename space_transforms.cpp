#include "space_transforms.h"
#include "space_filter.h"


void SpaceProcess(Mat& image, Mat& out_img)
{
	cout << "\nͼ��ߴ�Ϊ:" << image.size() << endl;
	cout << "\n�����б�:\n" << "0:���� 1:ƽ�� 2:���� 3:��ת 4:����\n" <<
		"5:�����˲� 6:��ֵ�˲� 7:��ֵ�˲� 8:��˹�˲�\n" <<
		"9:Roberts 10:Perwitt 11:Sobel 12:Laplacian" << endl;
	cout << "\nѡ�������Ϊ:";
	int mode = -1;
	cin >> mode;

	if (mode == 0) {
		int mode1 = -1;
		cout << "����1ʱ��ֱ���� 2ʱˮƽ����" << endl;
		cin >> mode1;
		double time0 = static_cast<double>(getTickCount());
		out_img = Mirror(image, mode1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 1) {
		int X_offset, Y_offset, mode1;
		cout << "��������X Y�Լ���Եģʽ(���Һ�����Ϊ�����򣬱�Եģʽ1Ϊ��ɫ 2Ϊ��ɫ)" << endl;
		cin >> X_offset >> Y_offset >> mode1;
		double time0 = static_cast<double>(getTickCount());
		out_img = Translate(image, X_offset, Y_offset, mode1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 2) {
		int start_x, start_y, width, height;
		cout << "�������뿪ʼx,y�����Լ����д�С" << endl;
		cin >> start_x >> start_y >> width >> height;
		double time0 = static_cast<double>(getTickCount());
		out_img = CutImage(image, start_x, start_y, width, height);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 3) {
		float angle; int mode1;
		cout << "����������ת�Ƕ�,�Լ���ֵ��ʽ(��ʱ��Ϊ������, ����1Ϊ����ڲ�ֵ 2Ϊ˫���Բ�ֵ)" << endl;
		cin >> angle >> mode1;
		double time0 = static_cast<double>(getTickCount());
		out_img = Rotate(image, angle, mode1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 4) {
		int width, height, mode1;
		cout << "�����������ź�ͼƬ���,�߶ȺͲ�ֵ����(1Ϊ����ڲ�ֵ 2λ˫���Բ�ֵ)" << endl;
		cin >> width >> height >> mode1;
		double time0 = static_cast<double>(getTickCount());
		if (mode1 == 1) out_img = NearestInterpolate(image, height, width);
		else if (mode1 == 2) out_img = BilinearInterpolate(image, height, width);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 5) {
		int kernel_size;
		cout << "�������˴�С(����)" << endl;
		cin >> kernel_size;
		double time0 = static_cast<double>(getTickCount());
		out_img = BoxFilter(image, kernel_size, false);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 6) {
		int kernel_size;
		cout << "�������˴�С(����)" << endl;
		cin >> kernel_size;
		double time0 = static_cast<double>(getTickCount());
		out_img = BoxFilter(image, kernel_size, true);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 7) {
		int kernel_size;
		cout << "�������˴�С(����)" << endl;
		cin >> kernel_size;
		double time0 = static_cast<double>(getTickCount());
		out_img = MedianFilter(image, kernel_size);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 8) {
		int kernel_size; double sigma;
		cout << "���������������˴�С(����)��sigma��ֵ" << endl;
		cin >> kernel_size >> sigma;
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(kernel_size, sigma);
		out_img = GaussFilter(image, gauss_kernel);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 9) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Roberts(image);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 10) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Prewitt_Sobel_Laplacian(image, 1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 11) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Prewitt_Sobel_Laplacian(image, 2);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
	else if (mode == 12) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Prewitt_Sobel_Laplacian(image, 3);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "ͼ�����ʱΪ:" << time0 << "��" << endl;
	}
}


//=========������===========//
Mat Mirror(Mat &image, int mode)  //modeΪ1ʱ��ֱ���� 2ʱˮƽ����
{
	Mat out_img = Mat::zeros(image.size(), image.type());
	int row_number = image.rows;
	int col_number = image.cols;
	int img_channels = image.channels();
	if (mode == 1)
	{
		for (int i = 0; i < row_number; i++)
		{
			uchar *data = image.ptr<uchar>(i);
			uchar *new_data = out_img.ptr<uchar>(row_number - i - 1);
			for (int j = 0; j < col_number; j++)
				for (int c = 0; c < image.channels(); c++)
					new_data[j * img_channels + c] = data[j * img_channels + c];
		}
	}
	else if (mode == 2)
	{
		for (int i = 0; i < row_number; i++)
		{
			uchar *data = image.ptr<uchar>(i);
			uchar *new_data = out_img.ptr<uchar>(i);
			for (int j = 0; j < col_number; j++)
				for (int c = 0; c < img_channels; c++)
					new_data[(col_number - j - 1) * img_channels + c] = data[j * img_channels + c];
		}
	}
	return out_img;
}

//==========ƽ�ƹ���=========//
Mat Translate(Mat &image, int X_offset, int Y_offset, int border_mode) //border_modeĬ������Ϊ1������ɫ��Ե�� ����Ϊ2������ɫ��Ե
{
	Mat out_img;
	if (border_mode == 1)
		out_img = Mat(image.size(), image.type(), Scalar(0, 0, 0));
	else if (border_mode == 2)
		out_img = Mat(image.size(), image.type(), Scalar(255, 255, 255));
	else
		cout << "�߽�ģʽ���ô���" << endl;
	int row_number = image.rows;
	int col_number = image.cols;
	int img_channel = image.channels();
	for (int i = 0; i < (row_number - Y_offset < row_number ? row_number - Y_offset : row_number); i++)
	{
		uchar *data = image.ptr<uchar>(i);
		uchar *new_data = out_img.ptr<uchar>(i + Y_offset < 0 ? 0 : i + Y_offset);
		for (int j = 0; j < (col_number - X_offset < col_number ? col_number - X_offset : col_number); j++)
			for (int c = 0; c < img_channel; c++)
				new_data[(j + X_offset < 0 ? 0 : j + X_offset)*img_channel + c] = data[j * img_channel + c];
	}
	return out_img;
}

//===========���й���===========//
Mat CutImage(Mat &image, int start_x, int start_y, int width, int height)
{
	if (start_x + width > image.cols || start_y + height > image.rows)
		cout << "��ȡ���򳬳�ͼ��߽�" << endl;
	Mat out_img = Mat::zeros(Size(width, height), image.type());
	int img_channel = image.channels();
	for (int i = 0; i < height; i++)
	{
		uchar *data = image.ptr<uchar>(start_y + i);
		uchar *new_data = out_img.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
			for (int c = 0; c < image.channels(); c++)
				new_data[j * img_channel + c] = data[(start_x + j) * img_channel + c];
	}
	return out_img;
}

//==========��ת����===========//
void BilinearInterpolate(Mat &image, Mat &out_image, int i, int j, double image_u, double image_v)	//��Ե���˫���Բ�ֵ
{
	int x0 = int(image_u), x1 = (int)(image_u + 0.5), y0 = (int)image_v, y1 = (int)(image_v + 0.5);
	double pu = fabs(image_u - x0), pv = fabs(image_v - y1);
	uchar *pix_00 = image.ptr<uchar>(y0, x0);
	uchar *pix_01 = image.ptr<uchar>(y0, x1);
	uchar *pix_10 = image.ptr<uchar>(y1, x0);
	uchar *pix_11 = image.ptr<uchar>(y1, x1);
	uchar *new_data = out_image.ptr<uchar>(i, j);
	for (int c = 0; c < image.channels(); c++)
		new_data[c] = (1 - pv)*(1 - pu)*pix_10[c] + (1 - pv)*pu*pix_11[c] +
		pv * (1 - pu)*pix_00[c] + pv * pu*pix_01[c];
}

void NearestInterpolate(Mat &image, Mat &out_image, int i, int j, double image_u, double image_v)
{
	//cout << i << " " << j << " " << image_u << " " << image_v << endl;
	int image_x = int(image_u + 0.5);
	int image_y = int(image_v + 0.5);
	image_y = min(image_y, image.rows - 1);
	image_x = min(image_x, image.cols - 1);
	//cout << out_image.size() << image.size() << image_y << " " << image_x << endl;
	uchar *data = image.ptr<uchar>(image_y, image_x);
	uchar *new_data = out_image.ptr<uchar>(i, j);
	for (int c = 0; c < image.channels(); c++)
		new_data[c] = data[c];
}

Mat Rotate(Mat &image, float angle, int interp_mode)  //interp_mode����Ϊ1��������ڲ�ֵ�� ����Ϊ2����˫���Բ�ֵ
{
	angle = angle / 180 * CV_PI;
	Mat out_img = Mat::zeros(Size(int(fabs(image.rows * sin(angle)) + fabs(image.cols * cos(angle)) + 0.5), int(fabs(image.rows * cos(angle)) + fabs(image.cols * sin(angle)) + 0.5)), image.type());
	Mat toPhysics = (Mat_<double>(3, 3) << 1, 0, -0.5*image.cols, 0, -1, 0.5*image.rows, 0, 0, 1);
	Mat rotate_matrix = (Mat_<double>(3, 3) << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1);
	Mat toPixel = (Mat_<double>(3, 3) << 1, 0, 0.5*out_img.cols, 0, -1, 0.5*out_img.rows, 0, 0, 1);
	Mat M_trans = toPixel * rotate_matrix * toPhysics;	//����ϵ�任
	Mat M_trans_inv = M_trans.inv();
	Mat out_uv(3, 1, CV_64F);
	out_uv.at<double>(2, 0) = 1;
	Mat image_uv(out_uv);
	double u_src = 0, v_src = 0;
	for (int i = 0; i < out_img.rows; ++i)
	{
		for (int j = 0; j < out_img.cols; ++j)
		{
			out_uv.at<double>(0, 0) = j;
			out_uv.at<double>(1, 0) = i;
			image_uv = M_trans_inv * out_uv;
			u_src = image_uv.at<double>(0, 0);
			v_src = image_uv.at<double>(1, 0);
			if (u_src < 0 || v_src < 0 || u_src > image.cols - 1 || v_src > image.rows - 1) continue;	//�������򲻽��в�ֵ
			if (interp_mode == 1) NearestInterpolate(image, out_img, i, j, u_src, v_src);
			else if (interp_mode == 2) BilinearInterpolate(image, out_img, i, j, u_src, v_src);
		}
	}
	return out_img;
}

//==========���Ź���===========//
Mat NearestInterpolate(Mat &image, int rows, int cols) //����ڲ�ֵ
{
	Mat resize_image = Mat::zeros(rows, cols, image.type());
	float row_ratio = float(image.rows) / float(rows);
	float col_ratio = float(image.cols) / float(cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int img_row = 0;
			int img_col = 0;
			img_row = cvRound(i * (float(image.rows) / rows));
			img_col = cvRound(j * (float(image.cols) / cols));
			img_row = min(img_row, image.rows - 1);
			img_col = min(img_col, image.cols - 1);
			uchar *data = image.ptr<uchar>(img_row, img_col);
			uchar *new_data = resize_image.ptr<uchar>(i, j);
			for (int c = 0; c < image.channels(); c++)
				new_data[c] = data[c];
		}
	}
	return resize_image;
}

Mat BilinearInterpolate(Mat &image, int rows, int cols) //˫���Բ�ֵ
{
	Mat resize_image = Mat::zeros(rows, cols, image.type());
	float row_ratio = float(image.rows) / float(rows);
	float col_ratio = float(image.cols) / float(cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int y0 = int(i * row_ratio);
			int x0 = int(j * col_ratio);
			int x1 = 0, y1 = 0;
			float u = 0, v = 0;
			if (y0 < image.rows - 1)
			{
				int y1 = y0 + 1;
				float u = i * row_ratio - y0;
			}
			else
			{
				y0 = image.rows - 1;
				int y1 = y0;
				float u = 0;
			}
			if (x0 < image.cols - 1)
			{
				int x1 = x0 + 1;
				float v = j * col_ratio - x0;
			}
			else
			{
				x0 = image.cols - 1;
				int x1 = x0;
				float v = 0;
			}
			uchar *pix_00 = image.ptr<uchar>(y0, x0);
			uchar *pix_01 = image.ptr<uchar>(y0, x1);
			uchar *pix_10 = image.ptr<uchar>(y1, x0);
			uchar *pix_11 = image.ptr<uchar>(y1, x1);
			uchar *new_data = resize_image.ptr<uchar>(i, j);
			for (int c = 0; c < image.channels(); c++)
				new_data[c] = (pix_01[c] - pix_00[c]) * v + (pix_10[c], pix_00[c]) * u
				+ (pix_11[c] + pix_00[c] - pix_01[c] - pix_10[c]) * u * v + pix_00[c];
		}
	}
	return resize_image;
}