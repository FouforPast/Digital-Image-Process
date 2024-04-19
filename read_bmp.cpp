#include "read_bmp.h"


Mat ReadData(ifstream &data, int rows, int cols, int offbits, int bit_count)
{
	Mat image;
	int cols_num = 0;
	//int colsDQ = 4 * ((cols * bit_count) / 32);	//数据对齐
	int colsDQ = 0;
	if (bit_count == 8) colsDQ = (cols + 3) / 4 * 4;
	else if (bit_count == 24) colsDQ = (cols * 3 + 3) / 4 * 4;
	if (bit_count == 8) //灰度图像
	{
		image = Mat::zeros(rows, cols, CV_8U);
		cols_num = cols;
	}
	else if (bit_count == 24)	//RGB24位彩色图像
	{
		image = Mat::zeros(rows, cols, CV_8UC3);
		cols_num = cols * 3;	//BGR3个字节
	}
	else
	{
		cout << "图片非8位或24位图像" << endl;
	}
	data.seekg(offbits, ios::beg);
	uchar *pData = new uchar[rows*colsDQ];
	data.read((char *)pData, rows*colsDQ);
	for (int i = rows - 1; i >= 0; --i)	//倒序存储
	{
		for (int j = 0; j < cols_num; j++)
		{
			//cout << *(pData + i * colsDQ + j);
			image.ptr<uchar>(rows - 1 - i)[j] = *(pData + i * colsDQ + j);
		}
	}
	delete[]pData;
	return image;
}

Mat ReadBmp()
{
	IMAGEFILEHEADER file_head;
	IMAGEINFOHEADER info_head;
	Mat image;
	string image_path; //图片地址
	cout << "输入8位或24位bmp图片地址:";
	cin >> image_path;
	ifstream data(image_path, ifstream::binary);
	data.seekg(ios::beg);
	if (!data)
	{
		cout << "图像读取失败";
	}
	data.read((char *)&file_head, sizeof(IMAGEFILEHEADER));	//读取位图文件头数据
	if (file_head.bfType != 'MB')
	{
		cout << "该数据不是bmp图片";
	}
	data.read((char *)&info_head, sizeof(IMAGEINFOHEADER));	//读取文件信息头数据
	image = ReadData(data, info_head.biHeight, info_head.biWidth, file_head.bfOffBits, info_head.biBitCount);
	return image;
}