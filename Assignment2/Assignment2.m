%===================================================
% Computer Vision Programming Assignment 2
%
% Student: Shirong Zheng
%
% City College of New York
%===================================================

% ---------------- Question 1 ------------------------
%    Generate the histogram of the image you are using, and then 
%    perform a number of histogram operations (such as contrast enhancement, 
%    thresholding and equalization) to make the image visually better for either
%    viewing or processing (10 points).  If it is a color image,  please first turn
%    it into an intensity image and then generate its histogram.  Try to display
%    your histogram (5 points), and make some observations of the image based on 
%    its histogram (5 points). What are the general distributions of the intensity
%    values? How many major peaks and valleys does your histogram have? How could
%    you use the histogram to understand, analyze or segment the image? Please also 
%    display the histograms of the processed images and provide a few important observations.
InputImage = 'IDPicture.bmp'; 
C1 = imread(InputImage);
[ROWS COLS CHANNELS] = size(C1);
CR1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CR1(:,:,band) = (C1(:,:,1));
end
CG1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CG1(:,:,band) = (C1(:,:,2));
end
CB1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CB1(:,:,band) = (C1(:,:,3));
end
Intensity=0.2999*CR1+0.587*CG1+0.114*CB1;
image(Intensity),title('Intensity Image');
figure();
I=(zeros(256,1));
for i= 1:250
    for j=1:250;
       I((Intensity(i,j)+1))=I((Intensity(i,j)+1))+1;
    end
end
x=1:256;
bar(x,I),title('Histogram')

% Histogram Equalization
GIm=imread(InputImage);
[x, y ,m]=size(GIm);
if m==3
    GIm=rgb2gray(GIm);
end
inf=whos('GIm');
Isize=0;
if inf.class=='uint8'
    Isize=256;
    else if inf.class=='uint68'
        Isize=65565;
        end
end
HIm=uint8(zeros(size(GIm,1),size(GIm,2)));
freq=zeros(256,1);
probf=zeros(256,1);
probc=zeros(256,1);
cum=zeros(256,1);
output=zeros(256,1);
freq=imhist(GIm);
sum=0;
no_bins=255;
probc=cumsum(freq)/numel(GIm);
output=round(probc*no_bins);
HIm(:)=output(GIm(:)+1);
figure
subplot(2,2,1)
imshow(GIm);
title('Original Image');
subplot(2,2,2)
imshow(HIm);
title('Image Equalization');
subplot(2,2,3)
imhist(GIm);
title('Original');
subplot(2,2,4)
imhist(HIm);
title('Histogram Equalization');
figure
subplot(2,2,1)
imshow(histeq(GIm));
title('Matlab Equalization');
subplot(2,2,2)
imshow(HIm);
title('Code Equalization');
subplot(2,2,3)
imhist(histeq(GIm));
title('Hist Matlab Equation');
subplot(2,2,4)
imhist(HIm);
title('Hist Code Equation');
figure
subplot(2,2,1)
imshow(GIm);
title('Original');
subplot(2,2,2)
imshow(HIm);
title('Image Equalization');
x=(1:Isize);
subplot(2,2,3)
plot(x,output);
title('Transform Function');
subplot(2,2,4)
plot(x,freq);
title('Transform Function');

%Thresholding
Thresholding = uint8(zeros(250, 250));
for i = 1:250
    for j = 1:250
        if Ins(i,j)<=110
            Thresholding(i,j) = 0;
        else
            Thresholding(i,j) = 255;
        end
    end
end
figure;
image(Thresholding),title('Thresholding T = 110');
colormap(MAP);
figure;
H2 = HIS(Thresholding, ROWS, COLS);
title('Histogram of Thresholind T = 110');
xlabel('pixels');

%Contrast Enhancement
figure
A=Intensity;
f0=0;f1=80;f2=180;f3=255; y0=0;y1=20;y2=240;y3=255; 
I=double(A); 
[m,n,o]=size(A) 
for a=1:m   
    for b=1:n     
        Q(a,b)=0;     
        if I(a,b)<80       
            Q(a,b)=I(a,b)*0.25;     
        elseif(I(a,b)>=80)&&(I(a,b)<180)       
            Q(a,b)=I(a,b)*2.2-156;      
        else          Q(a,b)=I(a,b)*0.2+204;     
        end
    end
end
P=mat2gray(Q); 
subplot(2,2,1);subimage(A);title('Original Graph'); 
subplot(2,2,2);subimage(P);title('Contrast Enhancement');  
subplot(2,2,3);plot([f0,f1,f2,f3],[y0,y1,y2,y3]);title('Grayscale Mapping Curve');

% ---------------- Question 2 ------------------------ 
%  Apply the 1x2 operator and Sobel operator to your image and analyze
%  the results of the gradient magnitude images (including vertical gradients, horizontal
%  gradients, and the combined) (10 points). Please don't forget to normalize your gradient
%  images, noting that the original vertical and horizontal gradients have both positive and 
%  negative values. I would recommend you to display the absolute values of the horizontal
%  and vertical gradient images.  Does the Sobel operator have any clear visual advantages 
%  over the 1x2 operator? Any disadvantages (5 points)? If you subtract the 1x2 edge image
%  from the Sobel are there any residuals? You might use two different types of images:
%  one ideal man-made image, and one image of a real scene with more details (5 points).
%  (Note: don't forget to normalize your results as shown in slide # 29 of  feature
%  extraction lecture: part 2)

%1x2 Operator 
one_mult_two_h = zeros(250, 250);
one_mult_two_v = zeros(250, 250);
index = [1, -1];
for i = 2:250
    for j = 2:250
        one_mult_two_h(i, j) = abs(Ins(i,j) - Ins(i-1, j));
        one_mult_two_v(i, j) = abs(Ins(i,j) - Ins(i, j-1));
    end
end

combine_h_v = uint8(zeros(250, 250));
for i = 1:250
    for j = 1:250
        combine_h_v(i, j) = sqrt((double(one_mult_two_h(i, j)).^2)+ (double(one_mult_two_v(i, j)).^2));
    end
end
figure;
Combine_One_Two = [one_mult_two_h, one_mult_two_v, combine_h_v];
image(Combine_One_Two),title('Combine Horizontal & Vertical Of 1x2 Operator');
colormap(MAP);

%Sobel Operator 
Sobel_Operator_x = (zeros(250, 250));
Sobel_Operator_y = (zeros(250, 250));
Gx = [1, 0, -1; 2, 0, -2; 1, 0, -1];
Gy = [-1, -2, -1; 0, 0 ,0; 1, 2, 1];
for i = 1: 250
    for j = 1: 250
        Sobel_Operator_x(i, j) = sobel(Ins, i, j, Gx)/4;
        Sobel_Operator_y(i, j) = sobel(Ins, i, j, Gy)/4;
    end
end
Combine_Sobel_x_y = uint8(zeros(250, 250));
for i = 1:250
    for j = 1:250
        Combine_Sobel_x_y(i, j) = sqrt((double(x_s_oper(i, j)).^2)+ (double(y_s_oper(i, j)).^2));
    end
end
figure;
Combine_Sobel = [Sobel_Operator_x, Sobel_Operator_y, Combine_Sobel_x_y];
image(Combine_Sobel),title('Combine Horizontal & Vertical Of Sobel Operator');
colormap(MAP);

subtr = zeros(250, 250);
for i = 1:250
    for j = 1:250
        subtr(i,j) = abs(Combine_Sobel_x_y(i,j) - Combine_One_Two(i,j));
    end
end
figure;
image(subtr),title('Sobel: 1x2 Operator');
colormap(MAP);

%Normalize Gradient Images
originalMinValue = double(min(min(Combine_Sobel)));
originalMaxValue = double(max(max(Combine_Sobel)));
originalRange = originalMaxValue - originalMinValue;
desiredMin = 0;
desiredMax = 255;
desiredRange = desiredMax - desiredMin;
dblImageS255 = desiredRange * (double(comb_sobel) - originalMinValue) / originalRange + desiredMin;
figure;
imshow(uint8(dblImageS255)),title('Normalize Gradient Images - Sobel Operator');
desiredMin = 0;
desiredMax = 1;
desiredRange = desiredMax - desiredMin;

ManMadeImage = 'dog.jpg'; 
C2 = imread(ManMadeImage);
figure;
image(C2),title('Original Man Made Image');
colormap(MAP);
ManMade_x = zeros(250, 250);
ManMade_y = zeros(250,250);
ManMade_Combine = zeros(250, 250);
for i = 1: 250
    for j = 1: 250
        ManMade_x(i, j) = sobel(C2, i, j, Gx)/4;
        ManMade_y(i, j) = sobel(C2, i, j, Gy)/4;
    end
end
for i = 1:250
    for j = 1:250
        ManMade_Combine(i,j) = sqrt((double(ManMade_x(i, j)).^2)+ (double(ManMade_y(i, j)).^2));
    end
end
Combine_ManMade = [ManMade_x, ManMade_y, ManMade_Combine];
figure;
image(Combine_ManMade),title('Ideal Man Made Image');
colormap(MAP);
 

% ---------------- Question 3 ------------------------
%   Generate edge maps of the above two combined gradient maps (10 points).  
%   An edge image should be a binary image with 1s as edge points and 0s as non-edge points. 
%   You may first generate a histogram of each gradient map,  and only keep certain percentage of pixels 
%   (e.g.  5% of the pixels with the highest gradient  values) as edge pixels (edgels) . Use the percentage
%   to automatically find a threshold for the gradient magnitudes. In your report, please write up the description
%   and probably equations for finding the threshold, and discuss if 5% is a good value. If not what is (5 points) ? 
%   You may also consider to use local, adaptive thresholds to different portions of the image so that all major 
%   edges will be shown up nicely (5 points). In the end, please try to generate a sketch of an image, such as the 
%   ID image of Prof. Zhu.
Generate_One_Two = uint8(zeros(250, 250));
Generate_Sobel_Operator = uint8(zeros(250,250));
One_Two_Thre = findThreshold(Combine_One_Two, 0.15);
Sobel_Thre = findThreshold(Combine_Sobel_x_y, 0.25);
for i = 1:250
    for j = 1:250
        if Combine_One_Two(i,j) <= One_Two_Thre
            Generate_One_Two(i,j) = 0;
        else
            Generate_One_Two(i, j) = 255;
        end
        if Combine_Sobel_x_y(i,j) <= Sobel_Thre
            Generate_Sobel_Operator(i,j) = 0;
        else
            Generate_Sobel_Operator(i, j) = 255;
        end
    end
end
two_generate = [Generate_One_Two, Generate_Sobel_Operator];
figure;
image(two_generate),title('Two Generate Edge Maps');
colormap(MAP);

%Local, Adaptive Thresholds
pieces = zeros(125,125,4);
pieces(:,:,1) = Combine_Sobel_x_y(1:125, 1:125);
pieces(:,:,2) = Combine_Sobel_x_y(126:250, 1:125);
pieces(:,:,3) = Combine_Sobel_x_y(1:125, 126:250);
pieces(:,:,4) = Combine_Sobel_x_y(126:250, 126:250);
piece_thre=zeros(4,1);
local_thre = zeros(125,125,4);
combine_four_local = zeros(250,250);
for i = 1:4
    piece_thre(i) = findThreshold(pieces(:,:,i), 0.20);
end
for i = 1:125
    for j = 1:125
        for k = 1:4
            if pieces(i,j,k) <= piece_thre(k)
                local_thre(i,j, k) = 0;
            else
                local_thre(i, j, k) = 255;
            end
        end
    end
end
combine_four_local(1:125, 1:125) = local_thre(:,:,1);
combine_four_local(126:250, 1:125) = local_thre(:,:,2);
combine_four_local(1:125, 126:250) = local_thre(:,:,3);
combine_four_local(126:250, 126:250) = local_thre(:,:,4);
figure;
image(combine_four_local),title('Seperating Into 4 Pieces And Thresholding');
colormap(MAP);

% ---------------- Question 4 ------------------------
%   What happens when you increase the size of the edge detection kernel from 1x2 to 3x3 and then to 5x5 ,
%   or 7x7? Discuss computational cost (in terms of members of operations, and the real machine running times - 5 points),
%   edge detection results (5 points) and sensitivity to noise, etc. (5 points). Note that your larger kernel should still
%   be an edge detector. Please list your kernels as matrices in your report, and tell us what they are good for (5 points). 
Gx5 = [1,2,0,-2,-1;4,8,0,-8,-4;6,12,0,-12,-6;4,8,0,-8,-4;1,2,0,-2,-1];
Gy5 = [-1,-4,-6,-4,-1; -2, -8, -12, -8, -2;0,0,0,0,0;2,8,12,8,2;1,4,6,4,1];
sobel_5x5 = zeros(250, 250);
sobel_5y5 = zeros(250,250);
sobel5_comb = zeros(250, 250);
for i = 1: 250
    for j = 1: 250
        sobel_5x5(i, j) = sobel(Ins, i, j, Gx5)/6;
        sobel_5y5(i, j) = sobel(Ins, i, j, Gy5)/6;
    end
end
for i = 1:250
    for j = 1:250
        sobel5_comb(i,j) = sqrt((double(sobel_5x5(i, j)).^2)+ (double(sobel_5y5(i, j)).^2));
    end
end
image_comb5 = [sobel_5x5,sobel_5y5,uint8(sobel5_comb)];
figure;
image(image_comb5),title('5x5 Sobel Operator');
colormap(MAP);

Gx7 = [3,2,1,0,-1,-2,-3;4,3,2,0,-2,-3,-4;5,4,3,0,-3,-4,-5;6,5,4,0,-4,-5,-6;5,4,3,0,-3,-4,-5;4,3,2,0,-2,-3,-4;3,2,1,0,-1,-2,-3];
Gy7 = [-3,-4,-5,-6,-5,-4,-3;-2,-3,-4,-5,-4,-3,-2;-1,-2,-3,-4,-3,-2,-1;0,0,0,0,0,0,0;1,2,3,4,3,2,1;2,3,4,5,4,3,2;,3,4,5,6,5,4,3];
sobel_7x7 = zeros(250, 250);
sobel_7y7 = zeros(250,250);
sobel7_comb = zeros(250, 250);
for i = 1: 250
    for j = 1: 250
        sobel_7x7(i, j) = sobel(Ins, i, j, Gx7)/8;
        sobel_7y7(i, j) = sobel(Ins, i, j, Gy7)/8;
    end
end
for i = 1:250
    for j = 1:250
        sobel7_comb(i,j) = sqrt((double(sobel_7x7(i, j)).^2)+ (double(sobel_7y7(i, j)).^2));
    end
end
image_comb7 = [sobel_7x7,sobel_7y7,uint8(sobel7_comb)];
figure;
image(image_comb7),title('7x7 Sobel Operator');
colormap(MAP);

% ---------------- Question 5 ------------------------
%    Suppose you apply the Sobel operator to each of the RGB color bands of a color image.  How might you combine
%    these results into a color edge detector (5 points)?  Do the resulting edge differ from the gray scale results?  How and
%    why (5 points)? You may compare the edge maps of the intensity image (of the color image), the gray-scale edge map that
%    are the combination of the three edge maps from three color bands, or a real color edge map that edge points have
%    colors (5 points). Please discuss their similarities and differences, and how each of them can be used for image 
%    enhancement or feature extraction (5 points). Note that you want to first generate gradient maps and then using 
%    thresholding to generate edge maps.  In the end, please try to generate a color sketch of an image, such as the
%    ID image of Prof. Zhu. You may also consider local, adaptive thresholding in generating a color edge map.

CR_x = zeros(250,250);
CR_y = zeros(250,250);
CG_x = zeros(250,250);
CG_y = zeros(250,250);
CB_x = zeros(250,250);
CB_y = zeros(250,250);
C_Combine = uint8(zeros(250, 250, 3));
for i = 1: 250
    for j = 1: 250
        CR_x(i,j) = sobel(C1(:,:,1), i, j, Gx);
        CR_y(i,j) = sobel(C1(:,:,1), i, j, Gy);
        CG_x(i,j) = sobel(C1(:,:,2), i, j, Gx);
        CG_y(i,j) = sobel(C1(:,:,2), i, j, Gy);
        CB_x(i,j) = sobel(C1(:,:,3), i, j, Gx);
        CB_y(i,j) = sobel(C1(:,:,3), i, j, Gy);
    end
end

ccx =zeros(250,250,3);
ccy =zeros(250,250,3);
ccx(:,:,1) = CR_x;
ccx(:,:,2) = CG_x;
ccx(:,:,3) = CB_x;
ccy(:,:,1) = CR_y;
ccy(:,:,2) = CG_y;
ccy(:,:,3) = CB_y;
for i = 1:250
    for j = 1:250
        for k = 1:3
            C_Combine(i,j,k) = sqrt((double(ccx(i, j, k))).^2+ (double(ccy(i, j, k))).^2);
        end
    end
end

figure;
cc_c = [ccx,ccy,C_Combine];
image(cc_c),title('Color Image Edge Detector');


c_thre = uint8(zeros(250,250,3));
holding = zeros(3,1);
for i = 1:3
    holding(i) = findThreshold(c_comb(:,:,i), 0.20);
end

for i = 1:250
    for j = 1:250
        for k = 1:3
            if C_Combine(i,j,k) <= holding(k)
                c_thre(i, j, k) = 0;
            else
                c_thre(i, j, k) = 255;
            end
        end
    end
end
figure;
image(c_thre), title('Thresholding Of Color Image');

%local, adaptive
r_pieces = zeros(125,125,4);
g_pieces = zeros(125,125,4);
p_pieces = zeros(125,125,4);
r_pieces(:,:,1) = C_Combine(1:125, 1:125, 1);
r_pieces(:,:,2) = C_Combine(126:250, 1:125, 1);
r_pieces(:,:,3) = C_Combine(1:125, 126:250, 1);
r_pieces(:,:,4) = C_Combine(126:250, 126:250, 1);
g_pieces(:,:,1) = C_Combine(1:125, 1:125, 2);
g_pieces(:,:,2) = C_Combine(126:250, 1:125, 2);
g_pieces(:,:,3) = C_Combine(1:125, 126:250, 2);
g_pieces(:,:,4) = C_Combine(126:250, 126:250, 2);
p_pieces(:,:,1) = C_Combine(1:125, 1:125, 3);
p_pieces(:,:,2) = C_Combine(126:250, 1:125, 3);
p_pieces(:,:,3) = C_Combine(1:125, 126:250, 3);
p_pieces(:,:,4) = C_Combine(126:250, 126:250, 3);
piece_thre = zeros(4,3);
local_thre = zeros(125,125,4,3);
combine_four_local = zeros(250,250,3);
for i = 1:4
    piece_thre(i,1) = findThreshold(r_pieces(:,:,i), 0.20);
    piece_thre(i,2) = findThreshold(g_pieces(:,:,i), 0.20);
    piece_thre(i,3) = findThreshold(p_pieces(:,:,i), 0.20);
end
for i = 1:125
    for j = 1:125
        for k = 1:4
            if r_pieces(i,j,k) <= piece_thre(k,1)
                local_thre(i, j, k, 1) = 0;
            else
                local_thre(i, j, k, 1) = 255;
            end
            if g_pieces(i,j,k) <= piece_thre(k,2)
                local_thre(i, j, k, 2) = 0;
            else
                local_thre(i, j, k, 2) = 255;
            end
            if p_pieces(i,j,k) <= piece_thre(k,3)
                local_thre(i, j, k, 3) = 0;
            else
                local_thre(i, j, k, 3) = 255;
            end
        end
    end
end
combine_four_local(1:125, 1:125, 1) = local_thre(:,:,1, 1);
combine_four_local(126:250, 1:125, 1) = local_thre(:,:,2, 1);
combine_four_local(1:125, 126:250, 1) = local_thre(:,:,3, 1);
combine_four_local(126:250, 126:250, 1) = local_thre(:,:,4, 1);
combine_four_local(1:125, 1:125, 2) = local_thre(:,:,1, 2);
combine_four_local(126:250, 1:125, 2) = local_thre(:,:,2, 2);
combine_four_local(1:125, 126:250, 2) = local_thre(:,:,3, 2);
combine_four_local(126:250, 126:250, 2) = local_thre(:,:,4, 2);
combine_four_local(1:125, 1:125, 3) = local_thre(:,:,1, 3);
combine_four_local(126:250, 1:125, 3) = local_thre(:,:,2, 3);
combine_four_local(1:125, 126:250, 3) = local_thre(:,:,3, 3);
combine_four_local(126:250, 126:250, 3) = local_thre(:,:,4, 3);
figure;
image(combine_four_local), title('Local Adaptive Thresholding');


% ---------------- Functions Part ------------------------
function output = findThreshold(image, percentage)
    histo = zeros(256,1);
    row = length(image(:,1));
    col = length(image(1, :));
    for i=1 : row
        for j=1:col
            histo(image(i,j)+1)= histo(image(i,j)+1) + 1;
        end
    end
    result = 0;
    threshold = 0;
    reach = row*col*(1-percentage);
    for i=1:256
        if result-reach >= 0 
            threshold = i-1;
            break;
        else
            result = result + histo(i);
        end
    end
    output = threshold;
end

function output = sobel(image, row, col, array)
    result = double(0);
    x = 1;
    y = 1;
    size = round(length(array(:,1))/2.5);
    for i = row-size : row + size
        for j = col-size: col + size
            if not (i<=0 || j<=0 || j>=251 || i >= 251)
                result = result + double(image(i, j)).*array(x, y);
            end
            y=y+1;
        end
        y=1;
        x=x+1;
    end
    output = abs(result);
end

function histo = HIS(image, row, col)
    pixels = zeros(256,1);
    for i=1 : row
        for j=1:col
            pixels(image(i,j)+1)= pixels(image(i,j)+1) + 1;
        end
    end
    histo = bar((1:256), pixels);
end
