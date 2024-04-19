#include "read_bmp.h"


Mat ReadData(ifstream &data, int rows, int cols, int offbits, int bit_count)
{
	Mat image;
	int cols_num = 0;
	//int colsDQ = 4 * ((cols * bit_count) / 32);	//���ݶ���
	int colsDQ = 0;
	if (bit_count == 8) colsDQ = (cols + 3) / 4 * 4;
	else if (bit_count == 24) colsDQ = (cols * 3 + 3) / 4 * 4;
	if (bit_count == 8) //�Ҷ�ͼ��
	{
		image = Mat::zeros(rows, cols, CV_8U);
		cols_num = cols;
	}
	else if (bit_count == 24)	//RGB24λ��ɫͼ��
	{
		image = Mat::zeros(rows, cols, CV_8UC3);
		cols_num = cols * 3;	//BGR3���ֽ�
	}
	else
	{
		cout << "ͼƬ��8λ��24λͼ��" << endl;
	}
	data.seekg(offbits, ios::beg);
	uchar *pData = new uchar[rows*colsDQ];
	data.read((char *)pData, rows*colsDQ);
	for (int i = rows - 1; i >= 0; --i)	//����洢
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
	string image_path; //ͼƬ��ַ
	cout << "����8λ��24λbmpͼƬ��ַ:";
	cin >> image_path;
	ifstream data(image_path, ifstream::binary);
	data.seekg(ios::beg);
	if (!data)
	{
		cout << "ͼ���ȡʧ��";
	}
	data.read((char *)&file_head, sizeof(IMAGEFILEHEADER));	//��ȡλͼ�ļ�ͷ����
	if (file_head.bfType != 'MB')
	{
		cout << "�����ݲ���bmpͼƬ";
	}
	data.read((char *)&info_head, sizeof(IMAGEINFOHEADER));	//��ȡ�ļ���Ϣͷ����
	image = ReadData(data, info_head.biHeight, info_head.biWidth, file_head.bfOffBits, info_head.biBitCount);
	return image;
}