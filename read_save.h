#pragma once
#include<fstream>
#include<windows.h>
#include<iostream>

using namespace std;

void printMenu();

//��ʾλͼ�ļ�ͷ��Ϣ   
void showBmpHead(BITMAPFILEHEADER pBmpHead);
//��ʾλͼ��Ϣͷ��Ϣ  
void showBmpInforHead(BITMAPINFOHEADER pBmpInforHead);
//����һ��ͼ��λͼ���ݡ����ߡ���ɫ��ָ�뼰ÿ������ռ��λ������Ϣ,����д��ָ���ļ���
bool readBmp2(char* bmpName);
//����ͼƬ
bool saveBmp(char* bmpName, unsigned char* imgBuf, int width, int height, int biBitCount, RGBQUAD* pColorTable); 