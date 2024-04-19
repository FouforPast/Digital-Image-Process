#pragma once
#pragma pack(2)	//���ýṹ����뷽ʽ
#include <stdio.h>
typedef struct tagIMAGEFILEHEADER {
	unsigned short    bfType;	//�ļ�����
	unsigned long	  bfSize;	//�ļ���С
	unsigned short    bfReserved1;
	unsigned short    bfReserved2;
	unsigned long     bfOffBits;	//λͼ����ƫ����
} IMAGEFILEHEADER;	//λͼ�ļ�ͷ

typedef struct tagIMAGEINFOHEADER {
	unsigned long      biSize;	//�ļ���Ϣͷ����
	long int		   biWidth;	//ͼ����
	long int           biHeight;	//ͼ��߶�
	unsigned short     biPlanes;	//λƽ����
	unsigned short     biBitCount;	//������ռλ��
	unsigned long      biCompression;	//�Ƿ�ѹ��
	unsigned long      biSizeImage;	//ͼ���С
	long int           biXPelsPerMeter;	//ˮƽ�ֱ���
	long int           biYPelsPerMeter;	//��ֱ�ֱ���
	unsigned long      biClrUsed;	//��ɫ��
	unsigned long      biClrImportant;	//��ʹ����Ҫ��ɫ��
} IMAGEINFOHEADER;	//�ļ���Ϣͷ