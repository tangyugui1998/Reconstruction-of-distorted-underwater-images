function [Iout,Tx,Ty]=bspline_transform_2d_double(Ox,Oy,Iin,dx,dy,mode)
% Bspline transformation grid function
% 
% [Iout,Tx,Ty]=bspline_transform_2d_double(Ox,Oy,Iin,dx,dy,mode)
%
% Inputs,
%   Ox, Oy : are the grid points coordinates
%   Iin : is input image, Iout the transformed output image
%   dx and dy :  are the spacing of the b-spline knots
%   mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            (cubic interpolation only supported by compiled mex file)
%            2: cubic interpolation and outsite pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
%
% Outputs,
%   Iout: The transformed image
%   Tx: The transformation field in x direction
%   Ty: The transformation field in y direction
%
% This function is an implementation of the b-spline registration
% algorithm in "D. Rueckert et al. : Nonrigid Registration Using Free-Form 
% Deformations: Application to Breast MR Images".
% 
% We used "Fumihiko Ino et al. : a data distrubted parallel algortihm for 
%  nonrigid image registration" for the correct formula's, because 
% (most) other papers contain errors. 
%
% Function is written by D.Kroon University of Twente (June 2009)


% Make polynomial look up tables 
Bu=zeros(4,dx);
Bv=zeros(4,dx);

x=0:dx-1;
u=(x/dx)-floor(x/dx);
Bu(0*dx+x+1) = (1-u).^3/6;
Bu(1*dx+x+1) = ( 3*u.^3 - 6*u.^2+ 4)/6;
Bu(2*dx+x+1) = (-3*u.^3 + 3*u.^2 + 3*u + 1)/6;
Bu(3*dx+x+1) = u.^3/6;


y=0:dy-1;
v=(y/dy)-floor(y/dy);
Bv(0*dy+y+1) = (1-v).^3/6;
Bv(1*dy+y+1) = ( 3*v.^3 - 6*v.^2 + 4)/6;
Bv(2*dy+y+1) = (-3*v.^3 + 3*v.^2 + 3*v + 1)/6;
Bv(3*dy+y+1) = v.^3/6;

% Make all x,y indices
[x,y]=ndgrid(0:size(Iin,1)-1,0:size(Iin,2)-1);

% Calculate the indexes need to loop up the B-spline values.
u_index=mod(x,dx); 
v_index=mod(y,dy);
            
i=floor(x/dx); % (first row outside image against boundary artefacts)
j=floor(y/dy);
        
% This part calculates the coordinates of the pixel
% which will be transformed to the current x,y pixel.

Tlocalx=0; Tlocaly=0;
for l=0:3,
    for m=0:3,
        IndexO1=i+l; IndexO2=j+m;
        Check_bound=(IndexO1<0)|(IndexO1>(size(Ox,1)-1))|(IndexO2<0)|(IndexO2>(size(Ox,2)-1));
        IndexO1(Check_bound)=1;
        IndexO2(Check_bound)=1;
        Check_bound_inv=double(~Check_bound);

        a=Bu(l*dx+u_index(:)+1);
        b=Bv(m*dy+v_index(:)+1);

        c=Ox(IndexO1(:)+IndexO2(:)*size(Ox,1)+1);
        Tlocalx=Tlocalx+Check_bound_inv(:).*a.*b.*c;
        
        c=Oy(IndexO1(:)+IndexO2(:)*size(Oy,1)+1);
        Tlocaly=Tlocaly+Check_bound_inv(:).*a.*b.*c;
    end
end

% All the neighborh pixels involved in linear interpolation.
xBas0=floor(Tlocalx); 
yBas0=floor(Tlocaly);
xBas1=xBas0+1;           
yBas1=yBas0+1;

% Linear interpolation constants (percentages)
xCom=Tlocalx-xBas0; 
yCom=Tlocaly-yBas0;
perc0=(1-xCom).*(1-yCom);
perc1=(1-xCom).*yCom;
perc2=xCom.*(1-yCom);
perc3=xCom.*yCom;

% limit indexes to boundaries
check_xBas0=(xBas0<0)|(xBas0>(size(Iin,1)-1));
check_yBas0=(yBas0<0)|(yBas0>(size(Iin,2)-1));
xBas0(check_xBas0)=0; 
yBas0(check_yBas0)=0; 
check_xBas1=(xBas1<0)|(xBas1>(size(Iin,1)-1));
check_yBas1=(yBas1<0)|(yBas1>(size(Iin,2)-1));
xBas1(check_xBas1)=0; 
yBas1(check_yBas1)=0; 

Iout=zeros(size(Iin));
for i=1:size(Iin,3);    
    Iin_one=Iin(:,:,i);
    % Get the intensities
    intensity_xyz0=Iin_one(1+xBas0+yBas0*size(Iin,1));
    intensity_xyz1=Iin_one(1+xBas0+yBas1*size(Iin,1)); 
    intensity_xyz2=Iin_one(1+xBas1+yBas0*size(Iin,1));
    intensity_xyz3=Iin_one(1+xBas1+yBas1*size(Iin,1));
    % Make pixels before outside Ibuffer mode
    if(mode==1||mode==3)
        intensity_xyz0(check_xBas0|check_yBas0)=0;
        intensity_xyz1(check_xBas0|check_yBas1)=0;
        intensity_xyz2(check_xBas1|check_yBas0)=0;
        intensity_xyz3(check_xBas1|check_yBas1)=0;
    end
    Iout_one=intensity_xyz0.*perc0+intensity_xyz1.*perc1+intensity_xyz2.*perc2+intensity_xyz3.*perc3;
    Iout(:,:,i)=reshape(Iout_one, [size(Iin,1) size(Iin,2)]);
end
    
% Store transformation fields
Tx=reshape(Tlocalx,[size(Iin,1) size(Iin,2)])-x;  Ty=reshape(Tlocaly,[size(Iin,1) size(Iin,2)])-y; 
    


