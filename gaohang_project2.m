clear all
clc

%Generates continuous function
f1=imread('bolts.jpg');  % 8 / 12
f2=imread('nuts.jpg');    % 17 /17
f3=imread('Mi_1.jpg');   % 26  / 27
f4=imread('Mi_2.jpg');   %29/ 39

%1. SEG_LAB_MOR operation
% fT is the binary image, Nt is the number of all objects.
[fT1, Nt1]= seg_lab_mor(f1);
% figure(1)
% imshow(fT1)
% title('nuts')
[fT2, Nt2] = seg_lab_mor(f2);
% figure(2)
% imshow(fT2)
% title('bolts')
[fT3, Nt3] = seg_lab_mor(f3);
% figure(3)
% imshow(fT3)
% title('Mix1')
[fT4, Nt4] = seg_lab_mor(f4);
Nt1
% figure(4)
% imshow(fT4)
% title('Mix2')

%2. Find the properties of bolts.jpg and nuts.jpg
Pt=regionprops(fT1,'Area','Perimeter');
At1=[Pt.Area];  
Pmt1=[Pt.Perimeter];
A1=mean(At1);
P1=mean(Pmt1);
U1=[A1; P1];

c1=[At1; Pmt1];
k1=[0 0 ;0 0];
for i=1:Nt1
    k1=k1+c1(:,i)*c1(:,i)';
end;
k1=1/Nt1*k1-U1*U1';

Pt=regionprops(fT2,'Area','Perimeter');
At2=[Pt.Area];
Pmt2=[Pt.Perimeter];
A2=mean(At2);
P2=mean(Pmt2);
U2=[A2; P2];

c2=[At2; Pmt2];
k2=[0 0 ;0 0];
for i=1:Nt2
    k2=k2+c2(:,i)*c2(:,i)';
end;
k2=1/Nt2*k2-U2*U2';

%2. Find fisher discriminant w1
w1=(k1+k2)^(-1)*(U1-U2);
%2. Find the threshold T1 of w1 discriminant
for i=1:Nt1
    D1(i)=w1'*c1(:,i);
end;
for i=1:Nt2
    D2(i)=w1'*c2(:,i);
end;
T1=(min(D1)+max(D2))/2;


%3. Find another discriminant w2
w2=[-1;1];
%3. Find the threshold T2 of w2 discriminant
for i=1:Nt1
    D1(i)=w2'*c1(:,i);
end;
for i=1:Nt2
    D2(i)=w2'*c2(:,i);
end;
T2=(max(D1)+min(D2))/2;

%4. Classify the objectives in Mix1.jpg by w1
Pt=regionprops(fT3,'Area','Perimeter');
At3=[Pt.Area];
Pmt3=[Pt.Perimeter];
c3=[At3; Pmt3];
%4. Find the bolts
[m,n]=size(fT3);
B3=zeros(m,n);
for i=1:Nt3
    D3(i)=w1'*c3(:,i);
    if D3(i)>T1
        A=(fT3==i);
        B3=B3+A;
    end
end
figure(5)
imshow(B3)
title('Classification.bolts of Mix1 by w1')
%4. Find the nuts
C3=zeros(m,n);
for i=1:Nt3
    D3(i)=w1'*c3(:,i);
    if D3(i)<=T1
        A=(fT3==i);
        C3=C3+A;
    end
end
figure(6)
imshow(C3)
title('Classification.nuts of Mix1 by w1')
%4. Classify the objectives in Mix2.jpg by w1
Pt=regionprops(fT4,'Area','Perimeter');
At4=[Pt.Area];
Pmt4=[Pt.Perimeter];
c4=[At4; Pmt4];
%4. Find the bolts
[m,n]=size(fT4);
B4=zeros(m,n);
for i=1:Nt4
    D4(i)=w1'*c4(:,i);
    if D4(i)>T1
        A=(fT4==i);
        B4=B4+A;
    end
end
figure(7)
imshow(B4)
title('Classification.bolts of Mix2 by w1')
%4. Find the nuts
C4=zeros(m,n);
for i=1:Nt3
    D4(i)=w1'*c4(:,i);
    if D4(i)<=T1
        A=(fT4==i);
        C4=C4+A;
    end
end
figure(8)
imshow(C4)
title('Classification.nuts of Mix2 by w1')

%4. Classify the objectives in Mix1.jpg by w2
Pt=regionprops(fT3,'Area','Perimeter');
At3=[Pt.Area];
Pmt3=[Pt.Perimeter];
c3=[At3; Pmt3];
%4. Find the bolts
[m,n]=size(fT3);
B3=zeros(m,n);
for i=1:Nt3
    D3(i)=w2'*c3(:,i);
    if D3(i)<=T2
        A=(fT3==i);
        B3=B3+A;
    end
end
figure(9)
imshow(B3)
title('Classification.bolts of Mix1 by w2')
%4. Find the nuts
C3=zeros(m,n);
for i=1:Nt3
    D3(i)=w2'*c3(:,i);
    if D3(i)>T2
        A=(fT3==i);
        C3=C3+A;
    end
end
figure(10)
imshow(C3)
title('Classification.nuts of Mix1 by w2')
%4. Classify the objectives in Mix2.jpg by w2
Pt=regionprops(fT4,'Area','Perimeter');
At4=[Pt.Area];
Pmt4=[Pt.Perimeter];
c4=[At4; Pmt4];
%4. Find the bolts
[m,n]=size(fT4);
B4=zeros(m,n);
for i=1:Nt4
    D4(i)=w2'*c4(:,i);
    if D4(i)<=T2
        A=(fT4==i);
        B4=B4+A;
    end
end
figure(11)
imshow(B4)
title('Classification.bolts of Mix2 by w2')
%4. Find the nuts
C4=zeros(m,n);
for i=1:Nt3
    D4(i)=w2'*c4(:,i);
    if D4(i)>T2
        A=(fT4==i);
        C4=C4+A;
    end
end
figure(12)
imshow(C4)
title('Classification.nuts of Mix2 by w2')

%let us try another two pictures 
f5=imread('lena_bw.tif');
[fT5, Nt5]= seg_lab_mor(f5);
%4. Classify the objectives in Mix1.jpg by w1
Pt=regionprops(fT5,'Area','Perimeter');
At5=[Pt.Area];
Pmt5=[Pt.Perimeter];
c5=[At5; Pmt5];
%4. Find the bolts
[m,n]=size(fT5);
B5=zeros(m,n);
for i=1:Nt5
    D5(i)=w1'*c5(:,i);
    if D5(i)>T1
        A=(fT5==i);
        B5=B5+A;
    end
end
figure(13)
imshow(B5)
title('Classification.bolts of lena by w1')
%4. Find the nuts
C5=zeros(m,n);
for i=1:Nt5
    D5(i)=w1'*c5(:,i);
    if D5(i)<=T1
        A=(fT5==i);
        C5=C5+A;
    end
end
figure(14)
imshow(C5)
title('Classification.nuts of lena by w1')
%4. Classify the objectives in lena by w2
%4. Find the bolts
[m,n]=size(fT5);
B5=zeros(m,n);
for i=1:Nt5
    D5(i)=w2'*c5(:,i);
    if D5(i)<=T2
        A=(fT5==i);
        B5=B5+A;
    end
end
figure(15)
imshow(B5)
title('Classification.bolts of lena by w2')
%4. Find the nuts
C6=zeros(m,n);
for i=1:Nt5
    D6(i)=w2'*c5(:,i);
    if D6(i)>T2
        A=(fT5==i);
        C6=C6+A;
    end
end
figure(16)
imshow(C6)
title('Classification.nuts of lena by w2')


