%% load ksp data
addPaths()
folder_names = {};
for ii = 1:1
    folder_names = cat(2, folder_names, ['scan' num2str(ii)]);
end

display(folder_names)

%%
for folder = folder_names
    
    display(folder)
    data_label = struct();
    data_label.folder = folder{1};

    if(strcmp(folder{1}, 'scan1'))
        data_path = ['data/' data_label.folder '/p1/e1/s1/kspace']; % scan1 has a special directory path...
    else
        data_path = ['data/' data_label.folder '/kspace'];
    end
    
    save_path = '../python/data2/';

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
    arc_mask = reshape(arc_mask, output_dims);

    %% avoid reformat, because tensorflow assumes a shape N x H x W x C normalize to be between 0, 1
    im_ref = sos(ifft3c(ksp), 4);

    norm_factor = max(abs(im_ref(:)));   
    ksp_to_save = ksp/norm_factor;

    %% save data
    save_filename = [data_label.folder];
    save_path_mask = [save_path save_filename '_mask']
    save_path_ref = [save_path save_filename '_ref']

    if(do_save)
        display('saving data')
        display('output sizes')
        size(arc_mask)
        size(ksp_to_save)
        writeReconData(save_path_mask, arc_mask);
        writeReconData(save_path_ref, ksp_to_save);
    end

    display('done ')
end