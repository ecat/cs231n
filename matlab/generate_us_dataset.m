%% load ksp data
addPaths()
data_path = 'data/scan2/kspace';

generate_real_data = true;

ksp = readReconData(data_path);
[nx, ny, nz, nc] = size(ksp);


%% do undersampling
Ry = 1;
Rz = 2;
calib_size = round(nz/20); % retain 5 percent of lines
calib_size = calib_size + mod(calib_size, 2) + 1; % ensure odd acs lines

output_dims = [1 ny nz 1];
arc_mask = get_arc_mask(ny, nz, Ry, Rz, calib_size, output_dims);

if(generate_real_data)
    display('generating real data')
    ksp_to_us = fft3c(abs(ifft3c(ksp)));
else
    display('generating complex data')
    ksp_to_us = ksp;
end

ksp_us = bsxfun(@times, ksp_to_us, arc_mask);


%% reformat into axial slices and normalize to be between 0, 1
im_us = ifft3c(ksp_us);
im_us = permute(im_us, [3 2 1 4]);

display(['imag to real ratio ' num2str(sum(abs(imag(im_us(:)))) / sum(abs(real(im_us(:)))))])

im_ref = sos(ifft3c(ksp), 4);
im_ref = permute(im_ref, [3 2 1 4]);

norm_factor = max(abs(im_ref(:)));

im_ref = im_ref / norm_factor; 
im_us = im_us / norm_factor;

figure; imshow3s([sos(im_us, 4), im_ref])

%% save data