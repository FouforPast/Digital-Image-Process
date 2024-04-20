 文件结构
- fdt.h 包含离散傅里叶正逆变换、离散余弦正逆变换、测试函数和相关辅助函数的声明
- fdt2.h 和fdt.h功能相同，只不过将fdt.h中的float*类型改为了complex<float>*类型，事实证明这个改动对计算性能影响不大
- filter.h 各种频域滤波器的实现和测试函数声明
- histogram.h 计算直方图，并实现直方图图像检索
- read_bmp.h 读取bmp图像文件的函数
- space_filter.h 各种空域滤波器
- space_transforms.h 空域变换
- DIP.cpp 程序主函数
- 其余cpp文件是对相应头文件的实现

DCT和DFT原理参见https://www.jianshu.com/p/13cd60f49286
