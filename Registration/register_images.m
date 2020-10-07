function [Ireg,O_trans,Spacing,M,Bx,By,Fx,Fy] = register_images(Imoving,Istatic,Options)
% This function register_images is the most easy way to register two
% images both affine and nonrigidly.
%
% Features:
% - It can be used with images from different type of scans or modalities.
% - It uses both a rigid transform and a nonrigid b-spline grid transform.
% - It uses grid refinement
% - It can be used with images of different sizes.
% - The function will automaticaly detect if the images can be registered
% with the sum of squared pixel distance (SSD), or when mutual information
% must be used as image similarity measure.
%
% Note: Compile the c-files with compile_c_files <enter> for more speed.
%
% [Ireg,Grid,Spacing,M,Bx,By,Fx,Fy] = register_images(Imoving,Istatic,Options);
%
% Inputs,
%   Imoving : The image which will be registerd
%   Istatic : The image on which Imoving will be registered
%   Options : Registration options, see help below
%
% Outputs,
%   Ireg : The registered moving image
%   Grid: The b-spline controlpoints, can be used to transform another
%       image in the same way: I=bspline_transform(Grid,I,Spacing);
%   Spacing: The uniform b-spline knot spacing
%	M : The affine transformation matrix
%   Bx, By : The backwards transformation fields of the pixels in
%       x and y direction seen from the  static image to the moving image.
%   Fx, Fy : The (approximated) forward transformation fields of the pixels in
%       x and y direction seen from the moving image to the static image.
%       (See the function backwards2forwards)
%
% Options,
%   Options.Similarity: Similarity measure (error) used can be set to:
%               sd : Squared pixel distance
%               mi : Normalized (Local) mutual information 
%               d, gd, gc, cc, pi, ld : see image_difference.m.
%   Options.Registration: 
%               Rigid    : Translation, Rotation
%               Affine   : Translation, Rotation, Shear, Resize
%               NonRigid : B-spline grid based local registration
%               Both     : Nonrigid and Affine (Default)
%   Options.Penalty: Thin sheet of metal smoothness penalty, default 1e-3,
%               if set to zero the registration will take a shorter time, but
%               will give a more distorted transformation field.
%   Options.Interpolation: Linear (default) or Cubic, the final result is
%               always cubic interpolated.
%   Options.MaxRef : Maximum number of grid refinements steps (default 2)
%   Options.Grid: Initial B-spline controlpoints grid, if not defined is initalized
%               with an uniform grid. (Used for in example while registering
%               a number of movie frames)
%   Options.Spacing: Spacing of initial B-spline grid in pixels 1x2 [sx sy]
%               sx and sy must be powers of 2, to allow grid refinement.
%   Options.MaskMoving: Image which is transformed in the same way as Imoving and
%               is multiplied with the individual pixel errors
%               before calculation of the te total (mean) similarity
%               error. In case of measuring the mutual information similariy
%               the zero mask pixels will be simply discarded.
%   Options.MaskStatic: Also a Mask but is used  for Istatic
%   Options.Verbose: Display Debug information 0,1 or 2
%
% Corresponding points / landmarks Options,
%   Options.Points1: List N x 2 of landmarks x,y in Imoving image
%   Options.Points2: List N x 2 of landmarks x,y in Istatic image, in which
%                     every row correspond to the same row with landmarks
%                     in Points1.
%   Options.PStrength: List Nx1 with the error strength used between the
%                     corresponding points, (lower the strenght of the landmarks
%                     if less sure of point correspondence).
%
%
% Example,
%   % Read two greyscale images of Lena
%   Imoving=imread('images/lenag1.png');
%   Istatic=imread('images/lenag2.png');
%
%   % Register the images
%   [Ireg,O_trans,Spacing,M,Bx,By,Fx,Fy] = register_images(Imoving,Istatic);
%
%   % Show the registration result
%   figure,
%   subplot(2,2,1), imshow(Imoving); title('moving image');
%   subplot(2,2,2), imshow(Istatic); title('static image');
%   subplot(2,2,3), imshow(Ireg); title('registerd moving image');
%   % Show also the static image transformed to the moving image
%   Ireg2=movepixels(Istatic,Fx,Fy);
%   subplot(2,2,4), imshow(Ireg2); title('registerd static image');
%
%  % Show the transformation fields
%   figure,
%   subplot(2,2,1), imshow(Bx,[]); title('Backward Transf. in x direction');
%   subplot(2,2,2), imshow(Fx,[]); title('Forward Transf. in x direction');
%   subplot(2,2,3), imshow(By,[]); title('Backward Transf. in y direction');
%   subplot(2,2,4), imshow(Fy,[]); title('Forward Transf. in y direction');
%
% % Calculate strain tensors
%   E = strain(Fx,Fy);
% % Show the strain tensors
%   figure,
%   subplot(2,2,1), imshow(E(:,:,1,1),[-1 1]); title('Strain Tensors Exx');
%   subplot(2,2,2), imshow(E(:,:,1,2),[-1 1]); title('Strain Tensors Exy');
%   subplot(2,2,3), imshow(E(:,:,2,1),[-1 1]); title('Strain Tensors Eyx');
%   subplot(2,2,4), imshow(E(:,:,2,2),[-1 1]); title('Strain Tensors Eyy');
%
% Example with Landmarks,
%
% % Read two images with triangles inside
%   Imoving=imread('images/landmarks1.png');
%   Istatic=imread('images/landmarks2.png');
%
% % Load the corresponding (matched) points
%   load('images/landmarks');
%
% % Register the images affine
%   Options = struct('Points1',Points1,'Points2',Points2,'PStrength',PStrength);
%   Ireg = register_images(Imoving,Istatic,Options);
%
%   % Show the start images
%   figure, imshow(Imoving+Istatic,[]);
%   % Show the registration result
%   figure, imshow(Ireg+Istatic,[]);
%
% Example Prostate,
% % Read two images 
%   Imoving=im2double(imread('images/prostate1.png'));
%   Istatic=im2double(imread('images/prostate2.png'));
%
% % Use mutual information
%   Options.Similarity='mi';
% % Set grid smoothness penalty
%   Options.Penalty = 1e-3;
%   Ireg = register_images(Imoving,Istatic,Options);
%  % Show the registration result
%   figure,
%   subplot(2,2,1), imshow(Imoving); title('moving image');
%   subplot(2,2,2), imshow(Istatic); title('static image');
%   subplot(2,2,3), imshow(Ireg); title('registerd moving image');
%
% Function is written by D.Kroon University of Twente (January 2010)

% add all needed function paths
add_function_paths;

% Disable warning
warning('off', 'MATLAB:maxNumCompThreads:Deprecated')

% Process inputs
defaultoptions=struct('Similarity',[],'Registration','Both','Penalty',1e-3,'MaxRef',2,'Grid',[],'Spacing',[],'MaskMoving',[],'MaskStatic',[],'Verbose',2,'Points1',[],'Points2',[],'PStrength',[],'Interpolation','Linear','Scaling',[1 1]);
if(~exist('Options','var')), Options=defaultoptions;
else
    tags = fieldnames(defaultoptions);
    for i=1:length(tags), if(~isfield(Options,tags{i})), Options.(tags{i})=defaultoptions.(tags{i}); end, end
    if(length(tags)~=length(fieldnames(Options))),
        warning('register_images:unknownoption','unknown options found');
    end
end

% Set parameters
type=Options.Similarity;
O_trans=Options.Grid; Spacing=Options.Spacing;
MASKmoving=Options.MaskMoving; MASKstatic=Options.MaskStatic;
Points1=Options.Points1; Points2=Options.Points2; PStrength=Options.PStrength;

% Start time measurement
if(Options.Verbose>0), tic; end

% Convert the input images to double with range 0..1
[Iclass,Imin,Imax,Imoving,Istatic]=images2double(Imoving,Istatic);
 
% Resize the moving image to fit the static image
[Istatic,Imoving,MASKmoving]=images2samesize(Istatic,Imoving,MASKmoving);

% Detect if the mutual information or pixel distance can be used as
% similarity measure. By comparing the histograms.
if(isempty(type)), type=check_image_modalities(Imoving,Istatic,Options); end

% Register the moving image affine to the static image
if(~strcmpi(Options.Registration(1),'N'))
     M=affine_registration(O_trans,Spacing,Options,Imoving,Istatic,MASKmoving,MASKstatic,type,Points1,Points2,PStrength);
else M=[]; 
end

% Make the initial b-spline registration grid
[O_trans,Spacing,MaxItt]=Make_Initial_Grid(O_trans,Spacing,Options,Imoving,M);
 
% Register the moving image nonrigid to the static image
if(strcmpi(Options.Registration(1),'N')||strcmpi(Options.Registration(1),'B'))
    [O_trans,Spacing]=nonrigid_registration(O_trans,Spacing,Options,Imoving,Istatic,MASKmoving,MASKstatic,type,Points1,Points2,PStrength,MaxItt);
end

% Transform the input image with the found optimal grid.
if ( nargout<5 )
    [Ireg]=bspline_transform(O_trans,Imoving,Spacing,3);
else
    [Ireg,Bx,By]=bspline_transform(O_trans,Imoving,Spacing,3);
end

% Make the forward transformation fields from the backwards
if ( nargout>6 ), [Fx,Fy]=backwards2forwards(Bx,By); end

% Convert the double registered image to the class and range of the input images
Ireg=Back2OldRange(Ireg,Iclass,Imin,Imax);

% End time measurement
if(Options.Verbose>0), toc, end

function add_function_paths()
try
    functionname='register_images.m';
    functiondir=which(functionname);
    functiondir=functiondir(1:end-length(functionname));
    addpath([functiondir '/functions'])
    addpath([functiondir '/functions_affine'])
    addpath([functiondir '/functions_nonrigid'])
catch me
    disp(me.message);
end

function [Istatic,Imoving,MASKmoving]=images2samesize(Istatic,Imoving,MASKmoving)
% Resize the moving image to fit the static image
if(sum(size(Istatic)-size(Imoving))~=0)
    Imoving = imresize(Imoving,[size(Istatic,1) size(Istatic,2)],'bicubic');
    if(~isempty(MASKmoving))
        MASKmoving = imresize(MASKmoving,[size(Istatic,1) size(Istatic,2)],'bicubic');
    end
end

function [Iclass,Imin,Imax,Imoving,Istatic]=images2double(Imoving,Istatic)
% Store the class of the inputs
Iclass=class(Imoving);

% Convert the inputs to double
Imoving=double(Imoving);
Istatic=double(Istatic);
Imin=min(min(Istatic(:)),min(Imoving(:))); Imax=max(max(Istatic(:)),max(Istatic(:)));
Imoving=(Imoving-Imin)/(Imax-Imin);
Istatic=(Istatic-Imin)/(Imax-Imin);

function Ireg=Back2OldRange(Ireg,Iclass,Imin,Imax)
% Back to old image range
Ireg=Ireg*(Imax-Imin)+Imin;

% Set the class of output to input class
if(strcmpi(Iclass,'uint8')), Ireg=uint8(Ireg); end
if(strcmpi(Iclass,'uint16')), Ireg=uint16(Ireg); end
if(strcmpi(Iclass,'uint32')), Ireg=uint32(Ireg); end
if(strcmpi(Iclass,'int8')), Ireg=int8(Ireg); end
if(strcmpi(Iclass,'int16')), Ireg=int16(Ireg); end
if(strcmpi(Iclass,'int32')), Ireg=int32(Ireg); end
if(strcmpi(Iclass,'single')), Ireg=single(Ireg); end

function M=affine_registration(O_trans,Spacing,Options,Imoving,Istatic,MASKmoving,MASKstatic,type,Points1,Points2,PStrength)
% Make smooth for fast affine registration
ISmoving=imfilter(Imoving,fspecial('gaussian',[10 10],2.5));
ISstatic=imfilter(Istatic,fspecial('gaussian',[10 10],2.5));

% Affine register the smoothed images to get the registration parameters
if(strcmpi(Options.Registration(1),'R'))
    if(Options.Verbose>0), disp('Start Rigid registration'); drawnow; end
    % Parameter scaling of the Translation and Rotation
    scale=[1 1 1];
    % Set initial affine parameters
    x=[0 0 0];
elseif(strcmpi(Options.Registration(1),'A'))
    if(Options.Verbose>0), disp('Start Affine registration'); drawnow; end
    % Parameter scaling of the Translation, Rotation, Resize and Shear
    scale=[1 1 0.01 0.01 0.01 0.01 0.01];
    % Set initial affine parameters
    x=[0 0 0 100 100 0 0];
elseif(strcmpi(Options.Registration(1),'B'))
    if(Options.Verbose>0), disp('Start Affine part of Non-Rigid registration'); drawnow; end
    % Parameter scaling of the Translation, Rotation, Resize and Shear
    scale=[1 1 0.01 0.01 0.01 0.01 0.01];
    % Set initial affine parameters
    x=[0 0 0 100 100 0 0];
else 
    warning('register_images:unknownoption','unknown registration method');
end

if(Options.Interpolation(1)=='L'), interpolation_mode=0; else interpolation_mode=2; end

% Register Affine with 3 scale spaces
for refine_itt=1:3
    if(refine_itt==1)
        ITmoving=imresize(ISmoving,0.25);
        ITstatic=imresize(ISstatic,0.25);
        Points1t=Points1*0.25; Points2t=Points2*0.25; PStrengtht=PStrength;
        if(~isempty(MASKmoving)), ITMASKmoving = imresize(MASKmoving,0.25); else ITMASKmoving=[]; end
        if(~isempty(MASKstatic)), ITMASKstatic = imresize(MASKstatic,0.25); else ITMASKstatic=[]; end
    elseif(refine_itt==2)
        x(1:2)=x(1:2)*2;
        ITmoving=imresize(ISmoving,0.5);
        ITstatic=imresize(ISstatic,0.5);
        Points1t=Points1*0.5; Points2t=Points2*0.5; PStrengtht=PStrength;
        if(~isempty(MASKmoving)), ITMASKmoving = imresize(MASKmoving,0.5); else ITMASKmoving=[]; end
        if(~isempty(MASKstatic)), ITMASKstatic = imresize(MASKstatic,0.5); else ITMASKstatic=[]; end
    elseif(refine_itt==3)
        x(1:2)=x(1:2)*2;
        ITmoving=Imoving;
        ITstatic=Istatic;
        ITMASKmoving = MASKmoving;
        ITMASKstatic = MASKstatic;
        Points1t=Points1; Points2t=Points2; PStrengtht=PStrength;
    end
    % Minimizer parameters

    % Use struct because expanded optimset is part of the Optimization Toolbox.
    optim=struct('GradObj','on','GoalsExactAchieve',1,'Display','off','StoreN',10,'HessUpdate','lbfgs','MaxIter',100,'MaxFunEvals',1000,'TolFun',1e-7,'DiffMinChange',1e-3);
    if(Options.Verbose>0), optim.Display='iter'; end
    x=fminlbfgs(@(x)affine_registration_error(x,scale,ITmoving,ITstatic,type,O_trans,Spacing,ITMASKmoving,ITMASKstatic,Points1t,Points2t,PStrengtht,interpolation_mode),x,optim);
end

% Scale the translation, resize and rotation parameters to the real values
x=x.*scale;

if(strcmpi(Options.Registration(1),'R'))
    % Make the rigid transformation matrix
    M=make_transformation_matrix(x(1:2),x(3));
else
    % Make the affine transformation matrix
    M=make_transformation_matrix(x(1:2),x(3),x(4:5),x(6:7));
end

function type=check_image_modalities(Imoving,Istatic,Options)
% Detect if the mutual information or pixel distance can be used as
% similarity measure. By comparing the histograms.
Hmoving=  hist(Imoving(:),60)./numel(Imoving);
Hstatic = hist(Istatic(:),60)./numel(Istatic);
if(sum(log(abs(Hmoving-Hstatic)+1))>0.5),
    type='mi';
    if(Options.Verbose>0), disp('Multi Modalities, Mutual information is used'); drawnow; end
else
    type='sd'; 
    if(Options.Verbose>0), disp('Same Modalities, Pixel Distance is used'); drawnow; end
end

function [O_trans,Spacing,MaxItt]=Make_Initial_Grid(O_trans,Spacing,Options,Imoving,M)
if(isempty(O_trans)),
    if(isempty(Options.Spacing))
        % Calculate max refinements steps
        MaxItt=min(floor(log2([size(Imoving,1) size(Imoving,2)]/4)));
        
        % set b-spline grid spacing in x and y direction
        Spacing=[2^MaxItt 2^MaxItt];
    else
        % set b-spline grid spacing in x and y direction
        Spacing=round(Options.Spacing);
        t=Spacing; MaxItt=0; while((nnz(mod(t,2))==0)&&(nnz(t<8)==0)), MaxItt=MaxItt+1; t=t/2; end
    end
    % Make the Initial b-spline registration grid
    if(strcmpi(Options.Registration(1),'N'))
        O_trans=make_init_grid(Spacing,[size(Imoving,1) size(Imoving,2)]);
    else
        O_trans=make_init_grid(Spacing,[size(Imoving,1) size(Imoving,2)],M);
    end
else
    MaxItt=0;
    TestSpacing=Spacing;
    while(mod(TestSpacing,2)==0), TestSpacing=TestSpacing/2; MaxItt=MaxItt+1; end

    if(~strcmpi(Options.Registration(1),'N'))
        % Calculate center of the image
        mean=size(Imoving)/2;
        % Make center of the image coordinates 0,0
        xd=O_trans(:,:,1)-mean(1); yd=O_trans(:,:,2)-mean(2);
        % Calculate the affine transformed coordinates
        O_trans(:,:,1) = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) * 1;
        O_trans(:,:,2) = mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) * 1;
    end
end
% Limit refinements steps to user input
if(Options.MaxRef<MaxItt), MaxItt=Options.MaxRef; end

function [O_trans,Spacing]=nonrigid_registration(O_trans,Spacing,Options,Imoving,Istatic,MASKmoving,MASKstatic,type,Points1,Points2,PStrength,MaxItt)
    % Non-rigid b-spline grid registration
    if(Options.Verbose>0), disp('Start non-rigid b-spline grid registration'); drawnow; end
    
    if (Options.Verbose>0), disp(['Current Grid size : ' num2str(size(O_trans,1)) 'x' num2str(size(O_trans,2))]); drawnow; end
    
    % set registration options.
    options.type=type;
    options.penaltypercentage=Options.Penalty;
    options.interpolation=Options.Interpolation;
    options.scaling=Options.Scaling;
    options.verbose=false;
 
    % Enable forward instead of central gradient incase of error measure is pixel distance
    if(strcmpi(type,'sd')), options.centralgrad=false; end
    
    % Reshape O_trans from a matrix to a vector.
    sizes=size(O_trans); O_trans=O_trans(:);
    
    % Make smooth images for fast registration without local minimums
    Hsize=round(0.25*(size(Imoving,1)/size(O_trans,1)+size(Istatic,2)/size(O_trans,2)));
    ISmoving=imfilter(Imoving,fspecial('gaussian',[Hsize Hsize],Hsize/5));
    ISstatic=imfilter(Istatic,fspecial('gaussian',[Hsize Hsize],Hsize/5));
    resize_per=2^(MaxItt-1);
    
    
    % Use struct because expanded optimset is part of the Optimization Toolbox.
    optim=struct('GradObj','on','GoalsExactAchieve',0,'StoreN',10,'HessUpdate','lbfgs','Display','off','MaxIter',100,'DiffMinChange',0.001,'DiffMaxChange',1,'MaxFunEvals',1000,'TolX',0.005,'TolFun',1e-8);
    if(Options.Verbose>0), optim.Display='iter'; end

    
    % No smoothing if no refinement
    if(MaxItt==0), ISmoving=Imoving; ISstatic=Istatic; resize_per=1; optim.TolX = 0.03; end
    
    % Resize the mask to the image size used in the registration
    if(~isempty(MASKmoving)), MASKmovingsmall=imresize(MASKmoving,1/resize_per);  else MASKmovingsmall=[]; end
    if(~isempty(MASKstatic)), MASKstaticsmall=imresize(MASKstatic,1/resize_per);  else MASKstaticsmall=[]; end
    
    
    % Start the b-spline nonrigid registration optimizer
    ISmoving_small=imresize(ISmoving,1/resize_per);
    ISstatic_small=imresize(ISstatic,1/resize_per);
    Spacing_small=Spacing/resize_per;
    Points1_small=Points1/resize_per;
    Points2_small=Points2/resize_per;
    PStrength_small=PStrength;
    O_trans = resize_per*fminlbfgs(@(x)bspline_registration_gradient(x,sizes,Spacing_small,ISmoving_small,ISstatic_small,options,MASKmovingsmall,MASKstaticsmall,Points1_small,Points2_small,PStrength_small),O_trans/resize_per,optim);
    % Reshape O_trans from a vector to a matrix
    O_trans=reshape(O_trans,sizes);

    for refine_itt=1:MaxItt
        if (Options.Verbose>0), disp('Registration Refinement'); drawnow; end
        
        % Refine the b-spline grid
        [O_trans,Spacing]=refine_grid(O_trans,Spacing,size(Imoving));
        
        % Make smooth images for fast registration without local minimums
        Hsize=round(0.25*(size(ISmoving,1)/size(O_trans,1)+size(ISstatic,2)/size(O_trans,2)));
        ISmoving=imfilter(Imoving,fspecial('gaussian',[Hsize Hsize],Hsize/5));
        ISstatic=imfilter(Istatic,fspecial('gaussian',[Hsize Hsize],Hsize/5));
        resize_per=2^(MaxItt-1-refine_itt);
        
        % No smoothing in last registration step
        if(refine_itt==MaxItt), ISmoving=Imoving; ISstatic=Istatic; resize_per=1; optim.TolX = 0.03; end
        
        if (Options.Verbose>0), disp(['Current Grid size : ' num2str(size(O_trans,1)) 'x' num2str(size(O_trans,2))]); drawnow; end
        
        % Reshape O_trans from a matrix to a vector.
        sizes=size(O_trans); O_trans=O_trans(:);
        
        % Resize the mask to the image size used in the registration
        if(~isempty(MASKmoving)), MASKmovingsmall=imresize(MASKmoving,1/resize_per);  else MASKmovingsmall=[]; end
        if(~isempty(MASKstatic)), MASKstaticsmall=imresize(MASKstatic,1/resize_per);  else MASKstaticsmall=[]; end
        
        % Start the b-spline nonrigid registration optimizer
        ISmoving_small=imresize(ISmoving,1/resize_per);
        ISstatic_small=imresize(ISstatic,1/resize_per);
        Spacing_small=Spacing/resize_per;
        Points1_small=Points1/resize_per;
        Points2_small=Points2/resize_per;
        PStrength_small=PStrength;
        O_trans = resize_per*fminlbfgs(@(x)bspline_registration_gradient(x,sizes,Spacing_small,ISmoving_small,ISstatic_small,options,MASKmovingsmall,MASKstaticsmall,Points1_small,Points2_small,PStrength_small),O_trans/resize_per,optim);
        
        
        % Reshape O_trans from a vector to a matrix
        O_trans=reshape(O_trans,sizes);
    end



