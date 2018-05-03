function [ arc_mask ] = get_arc_mask( ny, nz, Ry, Rz, calib_size, output_dims )
%GET_ARC_MASK makes an arc mask

    us_mask = get_us_mask(ny, nz, Ry, Rz, output_dims);
    calib_mask = get_calib_mask(ny, nz, calib_size, output_dims);
    
    arc_mask = us_mask | calib_mask;

end

