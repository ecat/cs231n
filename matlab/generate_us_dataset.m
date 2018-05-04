%% load ksp data
addPaths()
data_path = 'data/scan2/kspace';

ksp = readReconData(data_path);
[nx, ny, nz, nc] = size(ksp);


%% do undersampling
Ry = 1;
Rz = 2;
calib_size = 10;
output_dims = [1 ny nz 1];
arc_mask = get_arc_mask(ny, nz, Ry, Rz, calib_size, output_dims);

ksp_real = fft3c(abs(ifft3c(ksp)));
ksp_us = bsxfun(@times, ksp_real, arc_mask);


%% reformat into axial slices and normalize to be between 0, 1
im_us = ifft3c(ksp_us);
im_us = permute(im_us, [3 2 1 4]);

display(['imag to real ratio' num2str(sum(abs(imag(im_us(:)))) / sum(abs(real(im_us(:)))))])
im_us = im_us / max(abs(im_us(:))); 

%% save data