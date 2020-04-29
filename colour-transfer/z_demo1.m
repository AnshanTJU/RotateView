fprintf('  ... load images\n');
path ='/export/zfr/segment/srcImage/';
mask_path = '/export/zfr/segment/mask/';
bg_path = '/export/zfr/segment/bg.jpg';

imgname_lists=dir(fullfile(mask_path));
imgname_lists = imgname_lists(3:end);

savepath0='/export/zfr/segment/result0/';
savepath='/export/zfr/segment/result2/';
rng(0);
bg_img0 = imread(bg_path);

for j = 1:size(imgname_lists,1)
    mkdir([savepath, imgname_lists(j).name]);
    mkdir([savepath0, imgname_lists(j).name]);
    imgname_list=dir(fullfile([path, imgname_lists(j).name]));
    imgname_list = imgname_list(3:end);
    fprintf([path, imgname_lists(j).name]);
    fprintf('\n')
    fprintf(1,'total images:%d\n',size(imgname_list,1));
    src_name = [path, imgname_lists(j).name, '/', imgname_list(1).name];
    mask_name = [mask_path, imgname_lists(j).name, '/', imgname_list(1).name];
    I1 = imread(src_name);
    I11=I1(:,:,1);
    I12=I1(:,:,2);
    I13=I1(:,:,3);
    I1_mask = imread([mask_name(1:end-4),'.png']);
    bg_img = imresize(bg_img0,[size(I1,1),size(I1,2)]);
    bg_img1=bg_img(:,:,1);
    bg_img2=bg_img(:,:,2);
    bg_img3=bg_img(:,:,3);
    I11(I1_mask==0) = bg_img1(I1_mask==0);
    I12(I1_mask==0) = bg_img2(I1_mask==0);
    I13(I1_mask==0) = bg_img3(I1_mask==0);
    I1(:,:,1)=I11;
    I1(:,:,2)=I12;
    I1(:,:,3)=I13;
    
    I1 = double(I1)/255;
    str1 = [savepath, imgname_lists(j).name, '/', imgname_list(1).name];
    str0 = [savepath0, imgname_lists(j).name, '/', imgname_list(1).name];
    imwrite(I1,str1);
    imwrite(I1,str0);
    
    for i=2:size(imgname_list,1)
        src_name = [path, imgname_lists(j).name, '/', imgname_list(i).name];
        mask_name = [mask_path, imgname_lists(j).name, '/', imgname_list(i).name];
        I0 = imread(src_name);
        I01=I0(:,:,1);
        I02=I0(:,:,2);
        I03=I0(:,:,3);
        I0_mask = imread([mask_name(1:end-4),'.png']);
        bg_img = imresize(bg_img0,[size(I0,1),size(I0,2)]);
        bg_img1=bg_img(:,:,1);
        bg_img2=bg_img(:,:,2);
        bg_img3=bg_img(:,:,3);
        I01(I0_mask==0) = bg_img1(I0_mask==0);
        I02(I0_mask==0) = bg_img2(I0_mask==0);
        I03(I0_mask==0) = bg_img3(I0_mask==0);
        I0(:,:,1)=I01;
        I0(:,:,2)=I02;
        I0(:,:,3)=I03;
    
        I0 = double(I0)/255;
        
        IR_mkl = colour_transfer_MKL(I0,I1);
        str1 = [savepath, imgname_lists(j).name, '/', imgname_list(i).name];
        str0 = [savepath0, imgname_lists(j).name, '/', imgname_list(i).name];
        imwrite(IR_mkl,str1);
        imwrite(I0,str0);
    end
end
exit