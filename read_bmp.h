#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include "bmp_head.h"
using namespace cv;
using namespace std;

Mat ReadData(ifstream &data, int rows, int cols, int offbits, int bit_count);
Mat ReadBmp();