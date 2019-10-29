%Changed all but central parietal to NaN

subject_list = {'105' '106' '107' '904' '905' '906'};
num_subjects = length(subject_list);
for s = 1:num_subjects
    %Load the dataset according to the vector containing the subjects tags
    filepath = 'Bad_Trials_Removed/';
    EEG = pop_loadset('filename',[subject_list{s} '.set'],'filepath', filepath);
    %changed channel to NaN
    EEG.data([21 25 26:48], : , : ) = NaN;
    n = subject_list{1,s};
    filepathSaved = 'Frontal/';
    pop_saveset(EEG, 'filename', n , 'filepath', filepathSaved);
    
end 

