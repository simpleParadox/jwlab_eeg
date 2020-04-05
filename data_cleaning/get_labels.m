% FILEPATH = 'Y:\Members_Current\Jenn\EEG study\Imported data\cleaned\';
FILEPATH = '/Volumes/OFFCAMPUS/Jenn/Imported data/cleaned/';
SUBJECTS = {'927'};

for curr_subject = SUBJECTS
    EEG = pop_loadset('filename', char(append(curr_subject, '_cleaned_ml.set')), 'filepath', FILEPATH);
    label = zeros(size(EEG.epoch, 2), 1);
    for i = 1:size(EEG.epoch, 2)
        val = EEG.epoch(i).eventmffkey_cel;
        if ischar(val) 
            val = str2double(val);
          else   
            val = str2double(val{1,1});
        end
        label(i) = val;
    end
    fileID = fopen(char(append(FILEPATH, curr_subject ,'_labels.txt')),'w','n','UTF-8');
    fprintf(fileID,'%g ', label);
    fclose(fileID);
end
