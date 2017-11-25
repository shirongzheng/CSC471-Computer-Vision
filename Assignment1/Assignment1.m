%===================================================
% Computer Vision Programming Assignment 1
%
% Student: Shirong Zheng
%
% @Zhigang Zhu, 2003-2009
% City College of New York
%===================================================

% ---------------- Step 1 ------------------------
% Read in an image, get information
% type help imread for more information 

InputImage = 'NY.bmp'; 
%OutputImage1 = 'IDPicture_bw.bmp';

C1 = imread(InputImage);
[ROWS COLS CHANNELS] = size(C1);

% ---------------- Step 2 ------------------------
% If you want to display the three separate bands
% with the color image in one window, here is 
% what you need to do
% Basically you generate three "color" images
% using the three bands respectively
% and then use [] operator to concatenate the four images
% the orignal color, R band, G band and B band

% First, generate a blank image. Using "uinit8" will 
% give you an image of 8 bits for each pixel in each channel
% Since the Matlab will generate everything as double by default
CR1 =uint8(zeros(ROWS, COLS, CHANNELS));

% Note how to put the Red band of the color image C1 into 
% each band of the three-band grayscale image CR1
for band = 1 : CHANNELS,
    CR1(:,:,band) = (C1(:,:,1));
end

% Do the same thing for G
CG1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CG1(:,:,band) = (C1(:,:,2));
end

% and for B
CB1 =uint8(zeros(ROWS, COLS, CHANNELS));
for band = 1 : CHANNELS,
    CB1(:,:,band) = (C1(:,:,3));
end

% Whenever you use figure, you generate a new figure window 
No1 = figure;  % Figure No. 1

%This is what I mean by concatenation
disimg = [C1, CR1;CG1, CB1]; 

% Then "image" will do the display for you!
image(disimg);

% ---------------- Step 3 ------------------------
% Now we can calculate its intensity image from 
% the color image. Don't forget to use "uint8" to 
% covert the double results to unsigned 8-bit integers

I1    = uint8(round(sum(C1,3)/3));

% You can definitely display the black-white (grayscale)
% image directly without turn it into a three-band thing,
% which is a waste of memeory space

No2 = figure;  % Figure No. 2
image(I1);

% If you just stop your program here, you will see a 
% false color image since the system need a colormap to 
% display a 8-bit image  correctly. 
% The above display uses a default color map
% which is not correct. It is beautiful, though

% ---------------- Step 4 ------------------------
% So we need to generate a color map for the grayscale
% I think Matlab should have a function to do this,
% but I am going to do it myself anyway.

% Colormap is a 256 entry table, each index has three entries 
% indicating the three color components of the index

MAP =zeros(256, 3);

% For a gray scale C[i] = (i, i, i)
% But Matlab use color value from 0 to 1 
% so I scale 0-255 into 0-1 (and note 
% that I do not use "unit8" for MAP

for i = 1 : 256,  % a comma means pause 
    for band = 1:CHANNELS,
        MAP(i,band) = (i-1)/255;      
           
    end 
end

%call colormap to enfore the MAP
colormap(MAP);

% I forgot to mention one thing: the index of Matlab starts from
% 1 instead 0.

% Is it correct this time? Remember the color table is 
% enforced for the current one, which is  the one we 
% just displayed.

% You can test if I am right by try to display the 
% intensity image again:

No3 = figure; % Figure No. 3
image(I1);


% See???
% You can actually check the color map using 
% the edit menu of each figure window

% ---------------- Step 5 ------------------------
% Use imwrite save any image
% check out image formats supported by Matlab
% by typing "help imwrite
% imwrite(I1, OutputImage1, 'BMP');


% ---------------- Step 6 and ... ------------------------
% Students need to do the rest of the jobs from c to g.
% Write code and comments - turn it in both in hard copies and 
% soft copies (electronically)

%Part C
%Equation: I = 0.299R + 0.587G + 0.114B (the NTSC standard for luminance)
IntensityImage = (0.299 * CR1) + (0.587 * CG1) + (0.114 * CB1);
No4=figure;
image(IntensityImage);
title('Part C:Intensity Image');

%Part D
No5 = figure;
K = [4 16 32 64];
for i = 1:length(K)
    IntensityThresh = IntensityImage;
    intervals = round(linspace(1,256,K(i)));
    for j = 1:length(intervals)-1
        IntensityThresh(IntensityThresh > intervals(j) & IntensityThresh < intervals(j+1)) = intervals(j);
    end
    subplot(2, 2, i);
    image(IntensityThresh);
    t = strcat('Part D:K = ', num2str(K(i)));  
    title(t);
end


%Part E
No6=figure
IE2 =uint8(zeros(ROWS, COLS, CHANNELS));

%Part E: level K = 2:
% Any value less than 128(256/2), return 0.
for i = 1 : 250,
    for j = 1 : 250,
        for band = 1 : 3,
            if C1(i,j,band) < 128
                IE2(i,j,band) = 0;
            else
                IE2(i,j, band) = 255;
            end
        end
    end
end

%Part E: level K = 4:
% Any values less than 64(256/4), return 0.
IE4 =uint8(zeros(ROWS, COLS, CHANNELS));

for i = 1 : 250,
    for j = 1 : 250,
        for band = 1:3,
            if C1(i,j,band) < 64
                IE4(i,j,band) = 0;
            elseif C1(i,j,band) <128
                IE4(i,j,band) = 64;
            elseif C1(i,j,band) <192
                IE4(i,j,band)=128;
            else
                IE4(i,j,band) = 255;
            end
        end
    end
end

subplot(2, 2, 1);
imshow(IE2);
title('Part E:K=2');
subplot(2, 2, 2);
imshow(IE4);
title('Part E:K=4');


%Part F
%Just Use  I' =C ln (I+1) this logarithmic function to implement quantize the
%orginal image to K level color image

No7=figure
Rband1 = C1(:, :, 1);
Gband1 = C1(:, :, 2);
Bband1 = C1(:, :, 3);

c = 0.15;
Rband2 = c*log(1+double(Rband1));
Bband2 = c*log(1+double(Bband1));
Gband2 = c*log(1+double(Gband1));

LogImage= cat(3, Rband2, Gband2, Bband2);

subplot(2, 4, 1);
imshow(C1);
title('Original Image');
subplot(2, 4, 2);
imshow(LogImage);
title('Log Transformation');
subplot(2, 4, 3);
imshow(Rband1);
title('Red Band');
subplot(2, 4, 4);
imshow(Rband2);
title('Red Log Band');
subplot(2, 4, 5);
imshow(Gband1);
title('Green Band');
subplot(2, 4, 6);
imshow(Gband2);
title('Green Log Band');
subplot(2, 4, 7);
imshow(Bband1);
title('Blue Band');
subplot(2, 4, 8);
imshow(Bband2);
title('Blue Log Band');

