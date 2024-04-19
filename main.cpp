#include "space_transforms.h"
#include "space_filter.h"
#include "read_bmp.h"
#include <stdlib.h>

void SpaceProcess(Mat &image, Mat &out_img);

int main()
{
	while (1)
	{
		Mat image = ReadBmp(); //读取bmp图像
		Mat out_img;
		SpaceProcess(image, out_img);	//空域图像处理函数
		imshow("处理图像", out_img);
		imshow("原始图像", image);
		cout << "\n按ESC退出处理，按任意键继续" << endl;
		if (waitKey(0) != 27) {
			system("cls");
			destroyAllWindows();
			continue;
		}
		else break;
	}
}

void SpaceProcess(Mat &image, Mat &out_img)
{
	cout << "\n图像尺寸为:" << image.size() << endl;
	cout << "\n功能列表:\n" << "0:镜像 1:平移 2:剪切 3:旋转 4:缩放\n" <<
		"5:方框滤波 6:均值滤波 7:中值滤波 8:高斯滤波\n" <<
		"9:Roberts 10:Perwitt 11:Sobel 12:Laplacian" << endl;
	cout << "\n选择功能序号为:";
	int mode = -1;
	cin >> mode;

	if (mode == 0) {
		int mode1 = -1;
		cout << "输入1时垂直镜像 2时水平镜像" << endl;
		cin >> mode1;
		double time0 = static_cast<double>(getTickCount());
		out_img = Mirror(image, mode1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 1) {
		int X_offset, Y_offset, mode1;
		cout << "依次输入X Y以及边缘模式(向右和向下为正方向，边缘模式1为黑色 2为白色)" << endl;
		cin >> X_offset >> Y_offset >> mode1;
		double time0 = static_cast<double>(getTickCount());
		out_img = Translate(image, X_offset, Y_offset, mode1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 2) {
		int start_x, start_y, width, height;
		cout << "依次输入开始x,y坐标以及剪切大小" << endl;
		cin >> start_x >> start_y >> width >> height;
		double time0 = static_cast<double>(getTickCount());
		out_img = CutImage(image, start_x, start_y, width, height);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 3) {
		float angle; int mode1;
		cout << "依次输入旋转角度,以及插值方式(逆时针为正方向, 输入1为最近邻插值 2为双线性插值)" << endl;
		cin >> angle >> mode1;
		double time0 = static_cast<double>(getTickCount());
		out_img = Rotate(image, angle, mode1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 4) {
		int width, height, mode1;
		cout << "依次输入缩放后图片宽度,高度和插值方法(1为最近邻插值 2位双线性插值)" << endl;
		cin >> width >> height >> mode1;
		double time0 = static_cast<double>(getTickCount());
		if (mode1 == 1) out_img = NearestInterpolate(image, height, width);
		else if (mode1 == 2) out_img = BilinearInterpolate(image, height, width);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 5) {
		int kernel_size;
		cout << "输入卷积核大小(奇数)" << endl;
		cin >> kernel_size;
		double time0 = static_cast<double>(getTickCount());
		out_img = BoxFilter(image, kernel_size, false);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 6) {
		int kernel_size;
		cout << "输入卷积核大小(奇数)" << endl;
		cin >> kernel_size;
		double time0 = static_cast<double>(getTickCount());
		out_img = BoxFilter(image, kernel_size, true);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 7) {
		int kernel_size;
		cout << "输入卷积核大小(奇数)" << endl;
		cin >> kernel_size;
		double time0 = static_cast<double>(getTickCount());
		out_img = MedianFilter(image, kernel_size);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 8) {
		int kernel_size; double sigma;
		cout << "依次输入输入卷积核大小(奇数)与sigma数值" << endl;
		cin >> kernel_size >> sigma;
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(kernel_size, sigma);
		out_img = GaussFilter(image, gauss_kernel);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 9) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Roberts(image);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 10) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Prewitt_Sobel_Laplacian(image, 1);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 11) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Prewitt_Sobel_Laplacian(image, 2);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
	else if (mode == 12) {
		double time0 = static_cast<double>(getTickCount());
		Mat gauss_kernel = GaussKernel(3, 1);
		image = GaussFilter(image, gauss_kernel);
		out_img = Prewitt_Sobel_Laplacian(image, 3);
		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		cout << "图像处理耗时为:" << time0 << "秒" << endl;
	}
}