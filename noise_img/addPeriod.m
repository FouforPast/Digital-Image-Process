

% t=imread('band1.bmp');
% [m,n]=size(t);
% t_1=t;
% for i=1:m
% for j=1:n
% t_1(i,j)=t(i,j)+100*sin(50*i)+40*sin(50*j);
% end
% end
% imshow(t),title('原图');
% imwrite(t_1, 'band1_period.bmp');
% figure,imshow(t_1),title('加入周期噪声后');
% I2=log(abs(fftshift(fft2(t))));

I=imread('band1.bmp');
F=fftshift(fft2(I));
F=add(F,0,3);
I2=uint8(real(ifft2(ifftshift(F))));
% I2 = I2/max(max(I2));
% imwrite(I2, 'band1_period.bmp');
I2 = imread('band1_period.bmp');
figure(1);
F2=fftshift(fft2(I2));
imshow(log(abs(F2)), []);
figure(2);
imshow(I2);
figure(3);
imshow(log(abs(F)), []);
F=add(F2,1, 1000);
figure(4);
imshow(log(abs(F)), []);
figure(5);
I2=abs(ifft2(ifftshift(F)));
imshow(I2, []);


function noise=add(ori, val, num)
[m,n]=size(ori);
maxV=max(max(abs(ori)));
if(val==0)
    v=ori(round(m/2),round(n/2));
else
    v=0;
end
noise=ori;
R=100;
r=2;
for i = 1:num
    y=round((m/2)+R*sin(2*i*pi/num));
    x=round((n/2)+R*cos(2*i*pi/num));
    for row=y-r:y+r
        for col=x-r:x+r
            if((row-y)^2+(col-x)^2<r^2)
%                 noise(row,col)=sqrt(v)+sqrt(v)*1i;
%                 noise(row,col)=v+0i;
                  noise(row,col)=0+v*1i;
            end
        end
    end
end
end