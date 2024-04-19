

#include <iostream>
#include<opencv2/opencv.hpp>
#define _crt_secure_no_warnings 1
#include"fdt.h"
#include"filter.h"
#include"histogram.h"
#include"space_transforms.h"
#include"space_filter.h"
#include"read_bmp.h"
#include"read_save.h"
#include<stdlib.h>

using namespace cv;
using namespace std;


int main()
{
    string file;

    // 直方图检索
    //file = ".\\test\\lena.bmp";
    //mat img = imread(file);
    //string path = ".\\test";
    //mysearch(img, path, correl);
    //mysearch(img, path, chisqr);
    //mysearch(img, path, intersect);
    //mysearch(img, path, bhattacharyya);
	//return 0;
    
    // 频域变换测试
    file = ".\\test\\lena.bmp";
    bool naive = false;
    testDFT(file, naive);
    testDCT(file, naive);

    // 频域滤波测试
    //bool gray = false;
    //testfilter(file, "ilpf", 80, gray);
    //testfilter(file, "blpf", 80, gray, 5);
    //testfilter(file, "glpf", 80, gray);
    //testfilter(file, "elpf", 80, gray, 4);
    //testfilter(file, "tlpf", 80, gray, 1, 100);
    //testfilter(file, "ihpf", 40, gray);
    //testfilter(file, "bhpf", 40, gray, 5);
    //testfilter(file, "ghpf", 40, gray);
    //testfilter(file, "ehpf", 40, gray, 4);
    //testfilter(file, "thpf", 20, gray, 1, 40);

    // 周期噪声去除
    //file = ".\\noise_img\\band1_period.bmp";
    //int r = 2;
    //vector<notchpairs> pair = { notchpairs(272,382,r),  notchpairs(183,430,r),
    //    notchpairs(185,532,r),  notchpairs(357,430,r),notchpairs(359,532,r),notchpairs(270,580,r) };
    //testfilter(file, "inorthf", 40, true, 1, 0, &pair);
    //testfilter(file, "ibef", 99, true, 1, 101);


 //   // 空域相关
 //   while (1)
 //   {
 //       mat image = readbmp(); //读取bmp图像
 //       mat out_img;
 //       spaceprocess(image, out_img);	//空域图像处理函数
 //       imshow("处理图像", out_img);
 //       imshow("原始图像", image);
 //       cout << "\n按esc退出处理，按任意键继续" << endl;
 //       if (waitkey(0) != 27) {
 //           system("cls");
 //           destroyallwindows();
 //           continue;
 //       }
 //       else break;
 //   }


	//// 基本操作，直方图计算和均衡
	////while (1)
	////{
	////	printmenu();
	////}

	return 0;
}

//#include<iostream>
//#include<cmath>
//#include<windows.h>
//#include<stdlib.h>
//#include<opencv2/opencv.hpp>
//
//using namespace std;
//
//#pragma pack(2)	
//
//
//struct bitmapfileheader
//{
//    unsigned short bftype;
//    unsigned long bfsize;
//    unsigned short bfreserved1;
//    unsigned short bfreserved2;
//    unsigned long bfoffbits;
//};
//
//struct bitmapinfoheader
//{
//    unsigned long bisize;
//    long biwidth;
//    long biheight;
//    unsigned short biplanes;
//    unsigned short bibitcount;
//    unsigned long bicompression;
//    unsigned long bisizeimage;
//    long bixpelspermeter;
//    long biypelspermeter;
//    unsigned long biclrused;
//    unsigned long biclrimportant;
//};
//
//struct bmp
//{
//    bitmapfileheader fh;
//    bitmapinfoheader ih;
//    unsigned char* data;
//    bmp() {};
//};
//
//// 打印出bmp文件的信息
//void printbmp(bmp& bmp) {
//    printf("bmp文件大小：%dkb\n", bmp.fh.bfsize / 1024);
//    printf("保留字：%d\n", bmp.fh.bfreserved1);
//    printf("保留字：%d\n", bmp.fh.bfreserved2);
//    printf("实际位图数据的偏移字节数: %d\n", bmp.fh.bfoffbits);
//    printf("位图信息头:\n");
//    printf("信息头的大小:%d\n", bmp.ih.bisize);
//    printf("位图宽度:%ld\n", bmp.ih.biwidth);
//    printf("位图高度:%ld\n", bmp.ih.biheight);
//    printf("图像的位面数(位面数是调色板的数量,默认为1个调色板):%d\n", bmp.ih.biplanes);
//    printf("每个像素的位数:%d\n", bmp.ih.bibitcount);
//    printf("压缩方式:%d\n", bmp.ih.bicompression);
//    printf("图像的大小:%d\n", bmp.ih.bisizeimage);
//    printf("水平方向分辨率:%d\n", bmp.ih.bixpelspermeter);
//    printf("垂直方向分辨率:%d\n", bmp.ih.biypelspermeter);
//    printf("使用的颜色数:%d\n", bmp.ih.biclrused);
//    printf("重要颜色数:%d\n", bmp.ih.biclrimportant);
//}
//
//// 根据文件名读取bmp文件
//bmp readbmp(string filename) {
//    bmp bmp;
//    file* fp = fopen(filename.c_str(), "rb");
//    if (fp == nullptr) {
//        printf("打开%s失败\n", filename.c_str());
//        throw std::invalid_argument("can not find " + filename);
//    }
//    fread(&bmp.fh, sizeof(bmp.fh), 1, fp);
//    if (bmp.fh.bftype != 19778) {
//        printf("该文件不是bmp文件\n");
//    }
//    fread(&bmp.ih, sizeof(bmp.ih), 1, fp);
//    fseek(fp, bmp.fh.bfoffbits, seek_set);
//    unsigned int linelength = (bmp.ih.biwidth * (bmp.ih.bibitcount / 8) + 3) / 4 * 4;
//    bmp.data = new unsigned char[linelength * bmp.ih.biheight];
//    unsigned x = fread(bmp.data, sizeof(unsigned char), linelength * bmp.ih.biheight, fp);
//    printf("读出%d字节的数据\n", linelength * bmp.ih.biheight);
//    bmp.ih.bisizeimage = linelength * bmp.ih.biheight;
//    fclose(fp);
//    return bmp;
//}
//
//// 将bmp文件转化为mat
//cv::mat getmat(bmp& bmp)
//{
//    cv::mat img;
//    unsigned channel = bmp.ih.bibitcount / 8;
//    if (channel == 1) {
//        img = cv::mat(bmp.ih.biheight, bmp.ih.biwidth, cv_8u);
//    }
//    else {
//        img = cv::mat(bmp.ih.biheight, bmp.ih.biwidth, cv_8uc3);
//    }
//    uchar* ptr = (uchar*)img.data;
//    long rowlength = bmp.ih.bisizeimage / bmp.ih.biheight;
//    for (int i = bmp.ih.biheight - 1; i >= 0; --i) {
//        for (int j = 0; j < bmp.ih.biwidth * channel; ++j) {
//            *(ptr++) = bmp.data[i * rowlength + j];
//        }
//    }
//    return img;
//}
//
//int main(void)
//{
//    string filename = "d:\\users\\user\\pictures\\lena.bmp";
//    bmp bmp = readbmp(filename);
//    cv::mat img = getmat(bmp);
//    cv::imshow("bmp文件", img);
//    cv::waitkey(0);
//    cv::destroyallwindows();
//    printbmp(bmp);
//    return 0;
//}