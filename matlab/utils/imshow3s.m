function h=imshow3s(im,z,mag, plot_title)
% IMSHOW3S displays 3D images as a 2D slice in Handle Graphics figure
%    This function displays 3D images as 2D slices in the axial 
%    orientation. The user can use the mouse-scroll to scroll through
%    different slices. The window/level can be adjusted using the left
%    mouse button. Additionally, the window/level can be reset using
%    the right mouse button.
%
%    IMSHOW3S(I) displays grayscale image I. Initialize display
%    to show the center slices.
%
%    IMSHOW3S(I,Z) displays grayscale image I. The slice number is
%    set Z. If Z=[] or out of range, then default slice number is used.
%
%    IMSHOW3S(I,Z,MAG) displays grayscale image I. The slice number is
%    set S. The displayed image is magnified by MAG percent.
%    
%
% (c) Joseph Y Cheng (jycheng@mrsrl.stanford.edu) 2011
    
% SVN info: 
%     Date:     $Date: 2012-02-14 16:18:58 -0800 (Tue, 14 Feb 2012) $
%     Revision: $Revision: 193 $
%     Author:   $Author: jycheng $
%     Id:       $Id: imshow3s.m 193 2012-02-15 00:18:58Z jycheng $

    ha.figure = gcf;
    clf; % clear current figure.
    
    if (ndims(im) ~= 3 || ~isreal(sum(im(:))))
        error('Input must be a real 3D volume');
    end

    if (nargin < 2 || isempty(z) || z<1 || z>size(im,3))
        z = round(size(im,3)/2);        
    end
    if (nargin < 3)
        ha.mag = 100;
    else
        ha.mag = mag;
    end
    
    if (nargin < 4)
        plot_title = [];
    end
    
    %% Initialize maps
    ims = im(:,:,z);
    map = [min(ims(:)) max(ims(:))];
    ha.scale_wl = abs(diff(map))/1000;
    ha.map = map; 
    
    %% Setup figure and plot
    figure(ha.figure); hold off;
    ha.fh = imshow(im(:,:,z),ha.map,'InitialMagnification',ha.mag);  
    ha.a = gca;
    set(ha.figure,'pointer','crosshair');
    imshow3_plot(ha,im,z, plot_title);

    %% Setup output
    if (nargout==1)
        h = ha.figure;
    end
    
function imshow3_plot(ha,im,z, plot_title)
% IMSHOW3_PLOT Function that actually plots the data.
    
    %figure(ha.figure); 
    set(ha.fh,'CData',im(:,:,z));
    set(ha.a, 'CLim',ha.map); drawnow;
    
    if(isempty(plot_title))
        title(ha.a, sprintf('XY-slice: %d',z)); 
    else
        title(ha.a, sprintf('%s : %d', plot_title, z));
    end
    
    %% Re-init callback function with new values
    set(ha.figure,'WindowButtonDownFcn', {@int_callback,ha,im,z, plot_title});
    set(ha.figure,'WindowScrollWheelFcn',{@int_scroll_callback,ha,im,z, plot_title});

function int_scroll_callback(a,event,ha,im,z, plot_title)
% INT_SCROLL_CALLBACK callback functions that handles mouse scrolls.
%   This allows the user to scroll through slices using the mouse
%   wheel.
    imz = size(im,3);
    %temp = get(ha.a3,'VerticalScrollCount')
    dz = event.VerticalScrollCount; 
    if (abs(dz)>imz)
        dz = rem(dz,imz); 
    end
    if (imz<dz+z)
        z = dz+z-imz;
    elseif (1>dz+z)
        z = dz+z+imz;
    else
        z = dz+z;
    end
    imshow3_plot(ha,im,z, plot_title);

function int_callback(obj,a,ha,im,z, plot_title)
% INT_CALLBACK Main callback function that handles all functionality.
    %[imx,imy,imz] = size(im);
    
    x1 = round(get(ha.a,'CurrentPoint'));
    y1 = x1(1,2); x1 = x1(1,1);
   
    stype = get(obj,'SelectionType');
    if strcmp(stype,'normal')
        %% For window/level
        %dbdisp('normal selection');
        %if (0<x1 && x1<=imz && 0<y1 && y1<=imy)
            wl.x = x1; wl.y = y1;
        %end 
        if (exist('wl','var'))
            set(ha.figure,'WindowButtonMotionFcn', ...
                          {@int_motion_callback,ha,im,z, plot_title, wl});
            set(ha.figure,'WindowButtonUpFcn',{@int_release_callback,ha});
        end
    elseif strcmp(stype,'alt')
        %% Reset window/level
        %dbdisp(sprintf('Reset window/level: %d',z));
        ims = im(:,:,z);
        map = [min(ims(:)) max(ims(:))];
        %if (0<x1 && x1<=imz && 0<y1 && y1<=imy)
            ha.map = map;
        %end   
        imshow3_plot(ha,im,z, plot_title);
    end

function int_motion_callback(a,b,ha,im,z, plot_title, wl)
% INT_MOTION_CALLBACK Callback function used to set window/level.
    temp = round(get(ha.a,'CurrentPoint'));
    x1 = temp(1,1); y1 = temp(1,2);
    xdiff = x1-wl.x; ydiff = y1-wl.y;
    ha.map = modifymap(ha.map,xdiff,ydiff,ha.scale_wl);

    %% update plots
    imshow3_plot(ha,im,z, plot_title);
        
function int_release_callback(a,b,ha)
% INT_RELEASE_CALLBACK Callback function used to stop window/leveling.
    set(ha.figure,'WindowButtonMotionFcn','');
    
function int_radio_callback(obj,a,ha)
% INT_RADIO_CALLBACK Callback function used to switch cursor type.    
    selectobj = get(obj,'SelectedObject');
    if (ha.u0 == selectobj)
        set(ha.figure,'pointer','arrow');
    elseif(ha.u1 == selectobj)
        set(ha.figure,'pointer','arrow');
    else
        set(ha.figure,'pointer','crosshair');
    end
    
function nmap = modifymap(map,xdiff,ydiff,scale)
% MODIFYMAP Uses distance data to change the image map.
    [w,l] = map2wl(map);
    w = w+ydiff*scale; l = l+xdiff*scale;
    nmap = wl2map(w,l);

function [w,l] = map2wl(map)
% MAP2WL map values to window/level values.
    w = map(2)-map(1);
    l = (map(2)+map(1))/2;
    
function map = wl2map(w,l)
% WL2MAP window/level values to map values.
    map = [l-w/2 l+w/2];
