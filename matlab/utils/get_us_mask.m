function [ us_mask ] = get_us_mask( ny, nz, Ry, Rz, output_dims  )
%GET_US_MASK 
    if(nargin < 4)
        output_dims = -1;
    end
    
    us_mask = ones(ny, nz);
    
    if(Ry > 1)
        us_mask(1:Ry:end, :) = 0;
    end
    
    if(Rz > 1)
        us_mask(:, 1:Rz:end) = 0;
    end
    
    
    if(prod(output_dims) == ny * nz) 
        us_mask = reshape(us_mask, [1 ny nz 1]);
    end

end

