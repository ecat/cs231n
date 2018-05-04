%% script to load a knee image and see coil sensitivities
addPaths()
data_path = 'data/scan2/kspace';

ksp = readReconData(data_path);

%%
[nx, ny, nz, nc] = size(ksp);

im_coils = ifft3c(ksp);
im = sos(im_coils, 4);

figure;
imshow3s(im);

%% show all coil images
im_coils_1 = [];
im_coils_2 = [];

for ii = 1:nc/2
    im_coils_1 = cat(2, im_coils_1, abs(im_coils(:, :, :, ii)));
    im_coils_2 = cat(2, im_coils_2, abs(im_coils(:, :, :, ii + nc/2)));
end

figure; imshow3s(cat(1, im_coils_1, im_coils_2));

%% https://mrirecon.github.io/bart/examples.html
calib_mask = get_calib_mask(ny, nz, calib_size, [1 ny nz 1]);
ksp_calib = bsxfun(@times, ksp, calib_mask);

%%
[calib emaps] = bart('ecalib -r 20 -c 0.94', ksp_calib); % use svd thresh 0.94 for scan2, default for scan1

%% show all coil sensitivities from espirit
calib_1 = [];
calib_2 = [];

for ii = 1:nc/2
    calib_1 = cat(2, calib_1, abs(calib(:, :, :, ii)));
    calib_2 = cat(2, calib_2, abs(calib(:, :, :, ii + nc/2)));
end

calib_3_saggital = cat(1, calib_1, calib_2);
calib_3_axial = permute(cat(3, calib_1, calib_2), [2 3 1 4]);

figure; imshow3s(calib_3_saggital)
figure; imshow3s(calib_3_axial)