#pragma once
#pragma pack(2)	//设置结构体对齐方式
#include <stdio.h>
typedef struct tagIMAGEFILEHEADER {
	unsigned short    bfType;	//文件类型
	unsigned long	  bfSize;	//文件大小
	unsigned short    bfReserved1;
	unsigned short    bfReserved2;
	unsigned long     bfOffBits;	//位图数据偏移量
} IMAGEFILEHEADER;	//位图文件头

typedef struct tagIMAGEINFOHEADER {
	unsigned long      biSize;	//文件信息头长度
	long int		   biWidth;	//图像宽度
	long int           biHeight;	//图像高度
	unsigned short     biPlanes;	//位平面数
	unsigned short     biBitCount;	//像素所占位数
	unsigned long      biCompression;	//是否压缩
	unsigned long      biSizeImage;	//图像大小
	long int           biXPelsPerMeter;	//水平分辨率
	long int           biYPelsPerMeter;	//竖直分辨率
	unsigned long      biClrUsed;	//颜色数
	unsigned long      biClrImportant;	//所使用重要颜色数
} IMAGEINFOHEADER;	//文件信息头