%Bayram Muradov
%REFERENCES:
%1.https://www.mathworks.com/help/images/ref/superpixels.html
%2.https://ivrl.epfl.ch/research-2/research-current/ research-superpixels/
%3.https://www.mathworks.com/help/images/ref/gabor.html


%compile the slic0 function
mex slicomex.c

%number of images
num_img=21;

%structure to keep images
images = cell(num_img,1);
%structure to keep labels
labels = cell(num_img, 1);
%structure to keep num. of labels for each image
numlabels=cell(num_img, 1);
%structure to keep gabor responses
gabor_resp=cell(num_img, 1);
%reading images in data directory
imagefiles = dir('cs484_hw3_data/*.jpg');
nfiles = length(imagefiles);
for i=1:nfiles
   curr_name = imagefiles(i).name;
   mess_read=['** reading image : ', curr_name, ' **'];
   disp(mess_read);
   curr_img = imread(curr_name);
   curr_label=oversegment(curr_img, 248);
   images{i} = curr_img;
   labels{i} = curr_label{1};
   numlabels{i} = curr_label{2};
   mess_label=['== label count:', num2str(curr_label{2}), ' =='];
   disp(mess_label);
end

%FUNCTION CALLS HAPPEN FROM HERE
%disp_segs(images, labels, nfiles);
% gabor_resp=compute_gabor(images, nfiles);
% for i=1:5
%     disp_gabor(gabor_resp{i});
% end
%function for oversegmentation of the image

%get_crd(labels, nfiles);

%TYPE: 1=color, 2=texture, 3=both
merge_regions(labels, 1, images, numlabels, 6);

function os_image=oversegment(image, n) %n=248
[labels, numlabels] = slicomex(image,n);%numlabels=number of superpixels
os_image={labels, numlabels};
end

%displays the segmentation of images
function disp_segs(images, labels, nfiles)
for i=1:nfiles
    figure;
    BW = boundarymask(labels{i});
    imshow(imoverlay(images{i},BW,'cyan'),'InitialMagnification',67)
end
end



%gabor tf calculator for all images
function mags=compute_gabor(images, nfiles)
mags=cell(nfiles, 1);
gaborBank = gabor([4 8 12 16],[0 45 90 135]);
for i=1:nfiles
    img_gr=rgb2gray(images{i});
    gaborMag = imgaborfilt(img_gr,gaborBank);
    mags{i}=gaborMag;
end
end

%displays gabor tf results for 4 scales&ort.
function disp_gabor(gaborMag)
figure
subplot(4,4,1); %show all 16 responses
gaborBank = gabor([4 8 12 16],[0 45 90 135]);
    for p = 1:16
        subplot(4,4,p)
        imshow(gaborMag(:,:,p),[]);
        theta = gaborBank(p).Orientation;
        lambda = gaborBank(p).Wavelength;
        title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
    end
end

%merges the regions
%TYPE: 1=color, 2=texture, 3=both
function merge_regions(superpixels, type, images, numlabels, treshold)
%for each region, COMPUTE:
color_fv=cell(21, 1); %keeps color fv for each image
%texture_fv=cell(21, 1); %keeps texture fv for each image
cv_tracker=1;
img_tracker=1;
gv_tracker=1;
%for gabor
gv_mags=cell(21, 1);
gaborBank = gabor([4 8 12 16],[0 45 90 135]);

%computing feature vectors
for i=1:21
    %for each superpixel, keep array
    colors=cell(length(superpixels{i}), 1);
    tracker=1; %color fv tracker
    gtracker=1; %gabor fv tracker
    idx = label2idx(superpixels{i}); %get indx of superpixels
    curr_img=images{img_tracker};
    curr_rows=size(curr_img, 1);
    curr_cols=size(curr_img, 2);
    gabors=cell(length(superpixels{i}), 1);
    curr_gray=rgb2gray(images{1});
    for labelVal=1:(numlabels{i}-1)
        if (type==1 || type==3) %color features
            redIdx = idx{labelVal};
            greenIdx = idx{labelVal}+curr_rows*curr_cols;
            blueIdx = idx{labelVal}+2*curr_rows*curr_cols;
            avg_red=mean(curr_img(redIdx));
            avg_green=mean(curr_img(greenIdx));
            avg_blue=mean(curr_img(blueIdx));
%             disp(avg_red);
%             disp(avg_green);
%             disp(avg_blue);
            colors{tracker}={avg_red, avg_green, avg_blue};
            tracker=tracker+1;
        end
        %Computationally heavy to get over all images. takes appx: 90 mins
        if(type==2 || type ==3) %gabor features
            mess_label=['== label num:', num2str(labelVal), '**image num:', num2str(i), ' =='];
            disp(mess_label);
            grIdx=idx{labelVal};
            img_gr=(curr_gray(grIdx));
            gaborMag = imgaborfilt(img_gr,gaborBank);
            disp(gaborMag);
            gabors{gtracker}=gaborMag;
            gtracker=gtracker+1;
        end
    end
    if(type==1||type==3)
        color_fv{cv_tracker}=colors;
        cv_tracker=cv_tracker+1;
    end
    if(type==2)
        gv_mags{gv_tracker}=gabors;
        gv_tracker=gv_tracker+1;
        break;
    end
    img_tracker=img_tracker+1;

end

%merging step

%recompute the feature vectors
for i=1:21 %for each image in dataset
    if (type==1||type==3) %color fv
        curr_fv=color_fv{i};
        tracker=length(curr_fv);
        while tracker~=0
            for j=1:(length(curr_fv)-1)
                if(euclidianDistance(curr_fv{j}, curr_fv{j+1})<treshold)
                    fin_red=average(curr_fv{i}{1}, curr_fv{i+1}{1});
                    fin_green=average(curr_fv{i}{2}, curr_fv{i+1}{2});
                    fin_blue=average(curr_fv{i}{3}, curr_fv{i+1}{3});
                    %update color feature vector to new values=>avg. of
                    %similar regions.
                    curr_fv{j}={fin_red, fin_green, fin_blue};
                    curr_fv{j+1}={fin_red, fin_green, fin_blue};
                end
            end
            tracker=tracker-1;
        end
        color_fv{i}=curr_fv;
    end
    if(type==2|| type==3)
        curr_gv=gv_mags{i};
        tracker=length(curr_gv);
        idx = label2idx(superpixels{i});
        outputImage = zeros(size(images{i}),'like',images{i});
        numRows = size(images{i},1);
        numCols = size(images{i},2);
        while tracker~=0
            for j=1:(length(curr_gv)-1)
                if(euclidianDistance(curr_gv{j}, curr_gv{j+1})<treshold)
                    redIdx = idx{j};
                    greenIdx = idx{j}+numRows*numCols;
                    blueIdx = idx{j}+2*numRows*numCols;
                    %update output image to the avg. of similar regions.
                    outputImage(redIdx) = mean(images{i}(redIdx));
                    outputImage(greenIdx) = mean(images{i}(greenIdx));
                    outputImage(blueIdx) = mean(images{i}(blueIdx));
                end
            end
        end
         figure
         imshow(outputImage,'InitialMagnification',67)
         break;
    end
end

%obtain an output image
for k=1:21
    outputImage = zeros(size(images{k}),'like',images{k});
    idx = label2idx(superpixels{k});
    numRows = size(images{k},1);
    numCols = size(images{k},2);
    curr_vec=color_fv{k};
    %curr_img=images{k};
    for labelVal=1:(numlabels{k}-1)
        %mess=['**iteration image** :', k];
        %disp(mess);
        %disp(curr_vec{labelVal}{2});
        redIdx = idx{labelVal};
        greenIdx = idx{labelVal}+numRows*numCols;
        blueIdx = idx{labelVal}+2*numRows*numCols;
%         outputImage(redIdx)=mean(curr_img(redIdx));
%         outputImage(greenIdx)=mean(curr_img(greenIdx));
%         outputImage(blueIdx)=mean(curr_img(blueIdx));
        outputImage(redIdx) = (curr_vec{labelVal}{1});
        outputImage(greenIdx) = (curr_vec{labelVal}{2});
        outputImage(blueIdx) = (curr_vec{labelVal}{3});
    end
    figure
    imshow(outputImage,'InitialMagnification',67)
end
end


function avg=average(val1, val2)
    avg=(val1+val2)/2;
end

function dist=euclidianDistance(cv1, cv2)
tf2 = isempty(cv2);
if(~tf2)
    red1=str2double(cv1{1});
    red2=str2double(cv2{1});
    green1=str2double(cv1{2});
    green2=str2double(cv2{2});
    blue1=str2double(cv1{3});
    blue2=str2double(cv2{3});
    dist = sqrt( (red1-red2)^2 + (green1-green2)^2+(blue1-blue2)^2 );
else
    dist=0;
end
end
