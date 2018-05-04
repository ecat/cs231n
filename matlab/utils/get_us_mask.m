function [ us_mask ] = get_us_mask( ny, nz, Ry, Rz, output_dims  )
%GET_US_MASK 
    if(nargin < 5)
        output_dims = -1;
    end
    
    us_mask = zeros(ny, nz);
    
    if(Ry > 1 && Rz == 1)
        us_mask(1:Ry:end, :) = 1;
    elseif(Rz > 1 && Ry == 1)
        us_mask(:, 1:Rz:end) = 1;
    else
        us_mask(1:Ry:end, 1:Rz:end) = 1;
    end    
    
    if(prod(output_dims) == ny * nz) 
        us_mask = reshape(us_mask, [1 ny nz 1]);
    end

end

