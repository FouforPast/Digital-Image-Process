clc;
clear;
%����ͼ�񣬲�ת��Ϊdouble��
I=imread('D:\\window\\USAS\\22�＾��ҵ\\ͼ����\\������ҵ\\band3.bmp');
I_D=im2double(I);
%���ͼ���С
[M,N, C]=size(I_D);
 
%%%========================���ɸ�˹������==================================
a=0;
b=0.001;
N_Gau=a+sqrt(b)*randn(M,N); 
%���������ӵ�ͼ����
J_Gaussian=I_D+N_Gau;
imwrite(J_Gaussian, './band3_gaussian.bmp')
%������������
%%%======================������������======================================
a=0;
b=0.08;
B=1;
N_Ray1=a+b*raylrnd(B,M,N);
%���������ӵ�ͼ����
J_rayl=I_D+N_Ray1;
imwrite(J_rayl, './band3_raylrnd.bmp')
%%=====================����٤������=======================================
a=0;
b=0.04;
A=1;
B=2;
N_Gam=a+b*gamrnd(A,B,[M,N]);
%���������ӵ�ͼ����
J_Gamma=I_D+N_Gam;
imwrite(J_Gamma, './band3_gamma.bmp')
%%=====================����ָ������=======================================
a=0;
b=0.04;
mu=2;
N_exp=a+b*exprnd(mu,[M,N]);
J_exp=I_D+N_exp;
imwrite(J_exp, './band3_exp.bmp')
%%=====================���Ӿ��ȷֲ�����=======================================
a=0;
b=0.08;
A=0;
B=2;
N_unif=a+b*unifrnd(A,B,[M,N]);
J_unif=I_D+N_unif;
imwrite(J_unif, './band3_unif.bmp')
%%=====================���ӽ��ηֲ�����=======================================
%a=0.02;
%J_salt = imnoise(I_D,'salt',a);
%imwrite(J_salt, './band1_salt.bmp')

image=I_D;
[width,height,z]=size(image);

result2=image;

%k1��k2��Ϊ�ж��ٽ��
k1=0.02;
k2=0.5;
%rand(m,n)���������m��n�еľ���ÿ������Ԫ�ض���0-1֮��
%����k����0.2������С��k��Ԫ���ھ�����Ϊ1����֮Ϊ0
a1=rand(width,height)<k1;
a2=rand(width,height)<k2;
%�ϳɲ�ɫͼ��
t1=result2(:,:,1);
t2=result2(:,:,2);
t3=result2(:,:,3);
%�ֳɺڵ� �׵� ���
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

