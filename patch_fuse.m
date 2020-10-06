function frame_combine=patch_fuse(frames,indexLoops)
% INPUT:
%         frames����a*b*n matrix,a is picture hight,b is picture width,and n is the number of pictures.
% OUTPUT:
%         frame_combine����a*b matrix��it's a fusion figure.
global image_height;
global image_width;
global patch_height;
global patch_width;
global patch_h;
global patch_w;
global regPATH;

	frames_patches = cell(patch_h,patch_w);
     for i = 1:patch_h 
         for j = 1:patch_w 
             for kk = 1:size(frames,3) 
                frames_patches{i,j}{kk} =frames((i-1)*patch_height/2+1:(i-1)*patch_height/2+patch_height,(j-1)*patch_width/2+1:(j-1)*patch_width/2+patch_width,kk);
             end
         end
     end
	%%
    Istatic_patches = cell(patch_h,patch_w);
    for ii = 1:patch_w 
        for jj = 1:patch_h 
            temp_img = zeros(size(frames_patches{jj,ii}{1}));
            
            for kk = 1:size(frames,3)
                temp_img = temp_img + frames_patches{jj,ii}{kk};
            end
            Istatic_patches{jj,ii} = temp_img/size(frames,3);
            ssimval=zeros(size(frames,3),1);
            for kk=1:size(frames,3)%
                ssimval(kk) = ssim(frames_patches{jj,ii}{kk},Istatic_patches{jj,ii});
            end
            idx=kmeans(ssimval,2);
            num1=sum(idx==1);
            num2=sum(idx==2);
            sum1=0;
            sum2=0;
            for i=1:size(idx)
                if(idx(i)==1)
                    sum1=sum1+ssimval(i);
                else
                    sum2=sum2+ssimval(i);
                end
            end
            sum1=sum1/num1;
            sum2=sum2/num2;
            if sum1>sum2 
                num_selected=num1;
                a=1;
            else 
                num_selected=num2;
                a=2;
            end
                step=1;
                for i = 1:size(idx) 
                    if(idx(i)==a) 
                        frames_patches_selected{jj,ii}{step}=frames_patches{jj,ii}{i};
                        step=step+1;
                    end
                end
               
            temp_img = zeros(size(frames_patches{jj,ii}{1}));
            for kk = 1:num_selected 
                temp_img = temp_img+frames_patches_selected{jj,ii}{kk};
            end
            Istatic_patches{jj,ii} = temp_img/num_selected;
        end
    end
 
    mask_h_r = zeros(patch_height/2,patch_width);
    h=hann(patch_height/2);
    h(ceil(patch_height/4):patch_height/2)=2-h(ceil(patch_height/4):patch_height/2);
 
    h = h/2;
    for i=1:patch_width 
        mask_h_r(:,i)=h(:);
    end
    mask_h_l = 1-mask_h_r;
   
    mask_w_r = zeros(image_height,patch_width/2);
    h=hann(patch_width/2);
    h(ceil(patch_width/4):patch_width/2)=2-h(ceil(patch_width/4):patch_width/2);

    h = h/2;
    for i=1:image_height
        mask_w_r(i,:)=h(:);
    end
    mask_w_l = 1-mask_w_r;
    

    frame_combine_height = cell(patch_w);

    for ii = 1:patch_w 
        frame_combine = zeros(image_height,patch_width);
        frame_combine(1:patch_height/2,:)= Istatic_patches{1,ii}(1:patch_height/2,:);
        frame_combine(image_height-patch_height/2+1:image_height,:)...
            =Istatic_patches{patch_h,ii}(patch_height/2+1:patch_height,:);
        
        for jj = 1:patch_h-1 
            frame_combine(jj*patch_height/2+1:(jj+1)*patch_height/2,:) = ...
                Istatic_patches{jj,ii}(patch_height/2+1:patch_height,:).*mask_h_l+...
                Istatic_patches{jj+1,ii}(1:patch_height/2,:).*mask_h_r;
        end
        frame_combie_height{ii}=frame_combine; 
    end
    frame_combine=zeros(image_height,image_width);
    frame_combine(:,1:patch_width/2)= frame_combie_height{1}(:,1:patch_width/2);
    frame_combine(:,image_width-patch_width/2+1:image_width)=frame_combie_height{patch_w}(:,patch_width/2+1:patch_width);
    
    for ii = 1:patch_w-1
        frame_combine(:,ii*patch_width/2+1:(ii+1)*patch_width/2) = ...
            frame_combie_height{ii}(:,patch_width/2+1:patch_width).*mask_w_l+...
            frame_combie_height{ii+1}(:,1:patch_width/2).*mask_w_r;
    end
    disp('patches fusion completed'); 
end
