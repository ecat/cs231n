function [ calib_mask ] = get_calib_mask( ny, nz, calib_size, output_dims )
%GET_CALIB_MASK Summary of this function goes here
%   ny
%   nz
% calib_size
% output_dims = -1 or 0 to keep default
    calib_mask = zeros(ny, nz); 
    calib_mask(ny/2 - calib_size/2 + 1: ny/2 + calib_size/2, nz/2 - calib_size/2 + 1: nz/2 + calib_size/2) = 1;
    
    if(prod(output_dims) == ny * nz) 
        calib_mask = reshape(calib_mask, [1 ny nz 1]);
    end

end

