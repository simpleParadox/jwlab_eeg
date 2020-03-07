FILEPATH = '/Users/roxyk/Desktop/lab/cleaned';
FILEPATH_OUT = '/Users/roxyk/Desktop/lab/db/';

SUBJECTS = {'105', '106', '107', '109', '111', '904', '905', '906', '112', '909', '910', '115', '116', '912'};
addpath(genpath('./EEGLab'));

for curr_subject = SUBJECTS
    EEG = pop_loadset('filename', char(append(curr_subject, '_cleaned_ml.set')), 'filepath', FILEPATH);
    M = zeros(size(EEG.epoch, 2),3); %create a data matrix with size of [# of trails x 3]                          
    for i = 1:size(EEG.epoch, 2)
        cell_num = EEG.epoch(i).eventmffkey_cel;
        obs_num = EEG.epoch(i).eventmffkey_obs;
        if ischar(cell_num) 
            cell_num = str2double(cell_num);
          else   
            cell_num = str2double(cell_num{1,1});
        end
        
        if ischar(obs_num)
            obs_num = str2double(obs_num);
        else
            obs_num = str2double(obs_num{1,1});
        end
        
        M(i,1) = i;            % 1st column: trail index
        M(i,2) = cell_num;  % 2nd column: cel#
        M(i,3) = obs_num;  % 3rd column: obs#
    end
    
    file_name = char(append(curr_subject, '_trial_cell_obs.csv'));
    out_path = char(append(FILEPATH_OUT, file_name));
    fileID = fopen(out_path,'w','n','UTF-8');
    if fileID == -1
        error('Author:Function:OpenFile', 'Cannot open file');
    end
    fprintf(fileID, '%s,%s,%s\n', "trial_index", "cell", "obs");
    fprintf(fileID, '%g,%g,%g\n', M(1,1), M(1,2), M(1,3));
    temp = M(2:end, :, :).';      %transpose is important
    fprintf(fileID, '%g,%g,%g\n', temp(:));
    fclose(fileID);
end
