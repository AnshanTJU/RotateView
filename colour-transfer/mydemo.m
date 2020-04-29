fprintf('  ... load images\n');
path ='../ACACTS/forground_input2/0039/';
savepath='./results/';
I1 = double(imread([path '0000.jpg']))/255;
rng(0);
for i=1:37
	str1 = sprintf('%04d.jpg',i);
	name =[path str1];
	I0 = double(imread(name))/255;
tic;
	IR_mkl = colour_transfer_MKL(I0,I1);
toc
	str1 = sprintf('%04d.jpg',i);
	imwrite(IR_mkl,[savepath str1]);
end
exit
