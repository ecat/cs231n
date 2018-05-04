function [ calib_mask ] = get_calib_mask( ny, nz, calib_size, output_dims )
%GET_CALIB_MASK Summary of this function goes here
%   ny
%   nz
% calib_size, odd number, includes the center region
% output_dims = -1 or 0 to keep default
    if(nargin < 4)
        output_dims = -1;
    end
        
    
    % this mask ensures that sinc pattern is real, can verify using code
    % below
    % a = [zeros(15, 1); ones(11, 1); zeros(14, 1)];
    % b = fftc(a, 1);
    % figure; subplot(211); plot(abs(b)); subplot(212); plot(angle(b));%
    
    calib_mask = zeros(ny, nz);     
    calib_extent = floor(calib_size/2);
    
    calib_mask(ny/2 - calib_extent + 1: ny/2 + calib_extent + 1, nz/2 - calib_extent + 1: nz/2 + calib_extent + 1) = 1;
    
    if(prod(output_dims) == ny * nz) 
        calib_mask = reshape(calib_mask, [1 ny nz 1]);
    end

end

