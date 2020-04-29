fprintf('  ... load images\n');
% path ='/export/zfr/segment/srcImage/';
path ='/export/zfr/segment/result1/';
mask_path = '/export/zfr/segment/mask/';
bg_path = '/export/zfr/segment/bg.jpg';

imgname_lists=dir(fullfile(mask_path));
imgname_lists = imgname_lists(3:end);

% savepath='/export/zfr/segment/result2/';
savepath='/export/zfr/segment/result3/';
rng(0);
bg_img0 = imread(bg_path);

for j = 1:size(imgname_lists,1)
    mkdir([savepath, imgname_lists(j).name]);
    imgname_list=dir(fullfile([path, imgname_lists(j).name]));
    imgname_list = imgname_list(3:end);
    fprintf([path, imgname_lists(j).name]);
    fprintf('\n')
    fprintf(1,'total images:%d\n',size(imgname_list,1));
    src_name = [path, imgname_lists(j).name, '/', imgname_list(1).name];
    % mask_name = [mask_path, imgname_lists(j).name, '/', imgname_list(1).name];
    mask_name = [mask_path, imgname_lists(j).name, '/', imgname_lists(j).name,'_',imgname_list(1).name];
    I1 = imread(src_name);
    
    I1 = double(I1)/255;
    
    for i=2:size(imgname_list,1)
        src_name = [path, imgname_lists(j).name, '/', imgname_list(i).name];
        % mask_name = [mask_path, imgname_lists(j).name, '/', imgname_list(i).name];
        mask_name = [mask_path, imgname_lists(j).name, '/', imgname_lists(j).name,'_',imgname_list(i).name];
        I0 = imread(src_name);
    
        I0 = double(I0)/255;
        
        IR_mkl = colour_transfer_MKL(I0,I1);
        str1 = [savepath, imgname_lists(j).name, '/', imgname_list(i).name];
        imwrite(IR_mkl,str1);
    end
end
exit