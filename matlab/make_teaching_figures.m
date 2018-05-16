%%
close all; clear all;

%% Figure 1 K-Space example
addPaths()
data_path = 'data/scan2/kspace';

ksp = readReconData(data_path);
img = sos(ifft3c(ksp), 4);

slice_to_show = 120;
slice = img(:, :, slice_to_show);

%%
figure; 
subplot(121);
imagesc(slice); colormap gray; axis image; axis off;
subplot(122);
imagesc(log10(abs(fft2c(slice))));  colormap gray; axis image; axis off;
print('figures/figure1', '-dpng')

%% Figure 2 K-space masks for CS, PI
[nx, ny, nz, nc] = size(ksp);

addpath ../../../vat/sampling 
arc_mask = get_arc_mask(64, 64, 1, 2, 10, -1);
cpd_mask = genVDCPD_BART(ones(128, 128), 2, 2, nx, ny);

%%
figure; 
subplot(121);
imagesc(arc_mask); colormap gray; axis image; axis off;
subplot(122);
imagesc(cpd_mask(:, :, 1)); colormap gray; axis image; axis off;
print('figures/figure2', '-dpng')


%% Figure 3 PSFs for CS, PI

line_arc = arc_mask(64, :)';
line_cpd = cpd_mask(64, :, 1)';

psf_arc = abs(fftc(line_arc, 1));
psf_cpd = abs(fftc(line_cpd, 1));

figure;
plot(psf_arc, 'LineWidth', 2); hold on;
plot(psf_cpd, 'LineWidth', 2);
xlim([1 128])
legend('Uniform + Low Res', 'Random')
set(gca,'xticklabel',{[]}) 
set(gca,'yticklabel',{[]}) 
print('figures/figure3', '-dpng')

%% Figure 4 undersampled images
[nx, ny] = size(slice);

feasiblePoints = ones(nx, ny);

arc_mask = get_arc_mask(nx, ny, 1, 2, 10, -1);

cpd_mask = genVDCPD_BART(feasiblePoints, 2, 2, nx, ny);
cpd_mask = cpd_mask(:, :, 3);

%%
arc_us = ifft2c(fft2c(slice) .* arc_mask);
cpd_us = ifft2c(fft2c(slice) .* cpd_mask);

figure; 
%subplot(131); imagesc(abs(slice));colormap gray; axis image; axis off;
subplot(121); imagesc(abs(arc_us)); colormap gray; axis image; axis off;
subplot(122); imagesc(abs(cpd_us)); colormap gray; axis image; axis off;
print('figures/figure4', '-dpng')

%% Figure 5 coil image montage
coil_ims = ifft3c(ksp);

coil_slice_ims = abs(coil_ims(:, :, slice_to_show, :));
coil_slice_ims = coil_slice_ims/max(abs(coil_slice_ims(:)));

figure;
montage(coil_slice_ims, 'Size', [2 4])
print('figures/figure5', '-dpng')