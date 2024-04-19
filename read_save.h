#pragma once
#include<fstream>
#include<windows.h>
#include<iostream>

using namespace std;

void printMenu();

//显示位图文件头信息   
void showBmpHead(BITMAPFILEHEADER pBmpHead);
//显示位图信息头信息  
void showBmpInforHead(BITMAPINFOHEADER pBmpInforHead);
//给定一个图像位图数据、宽、高、颜色表指针及每像素所占的位数等信息,将其写到指定文件中
bool readBmp2(char* bmpName);
//保存图片
bool saveBmp(char* bmpName, unsigned char* imgBuf, int width, int height, int biBitCount, RGBQUAD* pColorTable); 