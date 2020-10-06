
clc;
clear ;
% close all;

addpath (genpath([pwd '\Registration\']));
addpath (genpath([pwd '\data sets\']));
addpath (genpath([pwd '\inexact_alm_rpca\']));

output_path = 'results\';
if ~exist(output_path,'dir')
    mkdir(output_path);
end
for i = 1:5
    output=[output_path num2str(i) '\'];
    if ~exist(output,'dir')
        mkdir(output);
    end
end 
%% Input
viariable_init();
frames=input_load();
[h,w,num]=size(frames);
Means=mean(frames,3);
Options = struct('Similarity','sd'); % use Options = struct('Similarity','mi') for mutual information (better but much slower)
%% Reconstruction
for indexLoops=1:5
     % Register the images to the current mean
    I_mean = Means(:,:,indexLoops);
    Istatic = patch_fuse(frames,indexLoops);
    t1 = clock;
    p = Istatic;
    r = 8;
    eps = 0.02^2;
    q = zeros(size(Istatic));
    q(:, :, 1) = guidedfilter(Istatic(:, :), p(:, :), r, eps);
    I_enhanced = (Istatic - q) * 5 + q;
    for indexImages=1:size(frames,3)
        Imoving=frames(:,:,indexImages);
        [reg_blur,O_trans,Spacing,M,Bx,By,Fx,Fy] = register_images(Imoving,I_enhanced,Options);
        % warp the original frame
        reg_noblur=bspline_transform(O_trans,Imoving,Spacing,3);
        frames(:,:,indexImages) = reg_noblur;
    end
    Means(:,:,indexLoops+1) = mean(frames,3);
    for j=1:num
        filename = [output_path,sprintf('%d',indexLoops),'\output',sprintf('%d',j),'.jpg']; 
        imwrite(frames(:,:,j),filename);
    end

end

for indexLoops=1:5
    filename = [output_path, 'mean',sprintf('%d',indexLoops),'.jpg']; 
    imwrite(Means(:,:,indexLoops),filename);
end

%% PCA
[a,b,m]=size(frames);
n=a*b;
D=zeros(n,m);
for i = 1 : m
    temp= frames(:,:,i);
    D(:,i)=reshape(temp,n,1);
end
% % % inexact_alm_rpca.m
[A_hat, E_hat ,iter] = inexact_alm_rpca(D);

if ~exist([output_path '\A_hat\'],'dir')
    mkdir([output_path '\A_hat\']);
end
if ~exist([output_path '\E_hat\'],'dir')
    mkdir([output_path '\E_hat\']);
end

imA=zeros(size(frames));
imE=zeros(size(frames));
for i=1:m
    %     A
    imA(:,:,i)=reshape(A_hat(:,i),[size(frames,1),size(frames,2)]);
    imwrite(imA(:,:,i),[[output_path '\A_hat\'] sprintf('Image_%.3d.bmp',i)],'bmp');
    
    %     E
    imE(:,:,i) = reshape(E_hat(:,i), [size(frames,1),size(frames,2)]);
    imwrite(imE(:,:,i),[[output_path '\E_hat\'] sprintf('Image_%.3d.bmp',i)],'bmp');
end
%% output
fusion_last = patch_fuse(imA,indexLoops+1);
output_filename = [output_path, 'Result.jpg']; 
imwrite(fusion_last,output_filename)


