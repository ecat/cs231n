%% load ksp data
addPaths()

data_label = struct();
data_label.folder = 'scan2';

data_path = ['data/' data_label.folder '/kspace'];
save_path = '../python/data/';

generate_real_data = true;
do_save = true;

%%
display('loading data')
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
    data_label.type = 'real';
    ksp_to_us = fft3c(abs(ifft3c(ksp)));
else
    display('generating complex data')
    data_label.type = 'complex';    
    ksp_to_us = ksp;
end

ksp_us = bsxfun(@times, ksp_to_us, arc_mask);


%% reformat into axial slices and normalize to be between 0, 1
im_us = ifft3c(ksp_us);
im_us = permute(im_us, [3 2 1 4]);

if (generate_real_data)
    ratio = sum(abs(imag(im_us(:)))) / sum(abs(real(im_us(:))));
    display(['imag to real ratio ' num2str(ratio)])
    assert(ratio < 1e-2)
end

im_ref = sos(ifft3c(ksp), 4);
im_ref = permute(im_ref, [3 2 1 4]);

norm_factor = max(abs(im_ref(:)));

im_ref = im_ref / norm_factor; 
im_us = im_us / norm_factor;

figure; imshow3s([sos(im_us, 4), im_ref])

%% save data
save_filename = [data_label.folder '_' data_label.type];
save_path_us = [save_path save_filename '_us']
save_path_ref = [save_path save_filename '_ref']

if(do_save)
    display('saving data')
    writeReconData(save_path_us, im_us);
    writeReconData(save_path_ref, im_ref);
end

