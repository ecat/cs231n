% script for generating comparison to conventional method
addPaths()

addpath /bmrNAS/people/pkllee/BMRrepo/Recon/gfactor/GRAPPA
addpath /bmrNAS/people/pkllee/projects/vat


%%
save_path = '../python/data/';
save_filename = 'scan1_real';
save_path_us = [save_path save_filename '_us'];
save_path_ref = [save_path save_filename '_ref'];

img_us = readReconData(save_path_us);
img_ref = readReconData(save_path_ref);

%%
[nx, ny, nz, nc] = size(img_us);

Ry = 1; Rz = 2;

output_dims = [1 ny nz 1];

calib_size = round(nz/20); % retain 5 percent of lines
calib_size = calib_size + mod(calib_size, 2) + 1; % ensure odd acs lines
calib_extent = floor(calib_size/2);

arc_mask = get_arc_mask(ny, nz, Ry, Rz, calib_size, output_dims);

ksp_us = bsxfun(@times, fft3c(img_us), arc_mask);

hybrid_data = ifftc(ksp_us, 1);
kcalib = hybrid_data(:, ny/2 - calib_extent + 1: ny/2 + calib_extent + 1, nz/2 - calib_extent + 1: nz/2 + calib_extent + 1, :);

%%
img_recon = zeros(size(img_ref));

for ii = 1:nx
    img_recon(ii, :, :) = reconstruct_GRAPPA(abs(hybrid_data(ii, :, :, :)) > 0, ...
        hybrid_data(ii, :, :, :), ...
        kcalib(ii, :, :, :), ...
        [1 5 5]);   
   
end

%%
img_recon_ax = permute(img_recon / mean(img_recon(:)), [2 3 1]);
img_ref_ax = permute(img_ref / mean(img_ref(:)), [2 3 1]);
figure; imshow3s(cat(2, img_recon_ax, img_ref_ax, abs(img_recon_ax - img_ref_ax)))