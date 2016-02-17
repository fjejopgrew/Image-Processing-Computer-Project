%k-means clustering method with 2 classes to segment the image
function [Dk, Nt]= seg_lab_mor(f)
fK = im2bw(f,0.4);
fK=~fK;
b=strel('square',6);
Dk=imopen(fK,b);
Dk=imclose(Dk,b);
[Dk, Nt]=bwlabel(Dk,8);
