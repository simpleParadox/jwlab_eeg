%Changed bad channels to NaN

subject_list = {'105' '106' '107' '904' '905' '906'};
num_subjects = length(subject_list);
for s = 1:num_subjects
    %Load the dataset according to the vector containing the subjects tags
    filepath = 'Bad_Trials_Removed/';
    EEG = pop_loadset('filename',[subject_list{s} '.set'],'filepath', filepath);
    %changed channel 61 to 64 to NaN
    EEG.data(61:64, : , : ) = NaN;
    n = subject_list{1,s};
    filepathSaved = 'Bad2_Channels_Removed/';
    pop_saveset(EEG, 'filename', n , 'filepath', filepathSaved);
    
end 

%%

%additional channels to NaN
filepath = 'Bad2_Channels_Removed/';
n = '904'; %ps id
EEG = pop_loadset('filename',[n '.set'],'filepath', filepath);
%changed channel 60 to NaN
EEG.data(16, : , : ) = NaN;
filepathSaved = 'Bad2_Channels_Removed/';
pop_saveset(EEG, 'filename', n , 'filepath', filepathSaved);

%additional channels to NaN
filepath = 'Bad2_Channels_Removed/';
n = '906'; %ps id
EEG = pop_loadset('filename',[n '.set'],'filepath', filepath);
%changed channel 60 to NaN
EEG.data(41, : , : ) = NaN;
filepathSaved = 'Bad2_Channels_Removed/';
pop_saveset(EEG, 'filename', n , 'filepath', filepathSaved);


%additional channels to NaN
filepath = 'Bad2_Channels_Removed/';
n = '905'; %ps id
EEG = pop_loadset('filename',[n '.set'],'filepath', filepath);
%changed channel 60 to NaN
EEG.data(51, : , : ) = NaN;
filepathSaved = 'Bad2_Channels_Removed/';
pop_saveset(EEG, 'filename', n , 'filepath', filepathSaved);