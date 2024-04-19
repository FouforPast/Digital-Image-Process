clc;
clear;
%读入图像，并转换为double型
I=imread('D:\\window\\USAS\\22秋季作业\\图像处理\\个人作业\\band3.bmp');
I_D=im2double(I);
%获得图像大小
[M,N, C]=size(I_D);
 
%%%========================生成高斯白噪声==================================
a=0;
b=0.001;
N_Gau=a+sqrt(b)*randn(M,N); 
%将噪声叠加到图像上
J_Gaussian=I_D+N_Gau;
imwrite(J_Gaussian, './band3_gaussian.bmp')
%保存噪声数据
%%%======================生成瑞利噪声======================================
a=0;
b=0.08;
B=1;
N_Ray1=a+b*raylrnd(B,M,N);
%将噪声叠加到图像上
J_rayl=I_D+N_Ray1;
imwrite(J_rayl, './band3_raylrnd.bmp')
%%=====================叠加伽马噪声=======================================
a=0;
b=0.04;
A=1;
B=2;
N_Gam=a+b*gamrnd(A,B,[M,N]);
%将噪声叠加到图像上
J_Gamma=I_D+N_Gam;
imwrite(J_Gamma, './band3_gamma.bmp')
%%=====================叠加指数噪声=======================================
a=0;
b=0.04;
mu=2;
N_exp=a+b*exprnd(mu,[M,N]);
J_exp=I_D+N_exp;
imwrite(J_exp, './band3_exp.bmp')
%%=====================叠加均匀分布噪声=======================================
a=0;
b=0.08;
A=0;
B=2;
N_unif=a+b*unifrnd(A,B,[M,N]);
J_unif=I_D+N_unif;
imwrite(J_unif, './band3_unif.bmp')
%%=====================叠加椒盐分布噪声=======================================
%a=0.02;
%J_salt = imnoise(I_D,'salt',a);
%imwrite(J_salt, './band1_salt.bmp')

image=I_D;
[width,height,z]=size(image);

result2=image;

%k1、k2作为判断临界点
k1=0.02;
k2=0.5;
%rand(m,n)是随机生成m行n列的矩阵，每个矩阵元素都在0-1之间
%这里k都是0.2，所以小于k的元素在矩阵中为1，反之为0
a1=rand(width,height)<k1;
a2=rand(width,height)<k2;
%合成彩色图像
t1=result2(:,:,1);
t2=result2(:,:,2);
t3=result2(:,:,3);
%分成黑点 白点 随机
t1(a1&a2)=0;
t2(a1&a2)=0;
t3(a1&a2)=0;
t1(a1& ~a2)=255;
t2(a1& ~a2)=255;
t3(a1& ~a2)=255;
result2(:,:,1)=t1;
result2(:,:,2)=t2;
result2(:,:,3)=t3;
imwrite(result2, 'band3_salt.bmp')

