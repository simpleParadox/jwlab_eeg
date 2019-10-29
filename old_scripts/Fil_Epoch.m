%% Code for filtering and epoching the data

subject_list = {'105' '106' '107' '904' '905' '906'};
num_subjects = length(subject_list);

for s = 1:num_subjects
    %Load the dataset according to the vector containing the subjects tags
    filepath = 'Boundaries_Removed/';
    EEG = pop_loadset('filename',[subject_list{s} '_BoundariesRemoved.set'],'filepath', filepath);
    EEG.setname = [subject_list{s} '_EEG'];
    EEG = pop_rmbase( EEG, [], []); %Remove the baseline of each channel
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    
    EEG = pop_eegfiltnew( EEG, 0.1, 50); %Bandpass filter: 0.1-50Hz
    %EEG = pop_eegfiltnew( EEG, 60.5, 59.5); %Notch filter: 59.5-60.5Hz

    %Save the filtered data into a new dataset
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', [subject_list{s} '_filtered']);
    
    %Extract epochs locked to the event "Wait", from 200ms before and 1200ms after
    EEG = pop_epoch( EEG, { 'Wait' }, [-0.2 1.2], 'newname', [subject_list{s} '_filtered_epochs'], 'epochinfo', 'yes');
    %Create a new dataset with the epochs
    [ALLEEG EEG CURRENTSET] = 	(ALLEEG, EEG, CURRENTSET, 'setname', [subject_list{s} '_filtered_epochs']);
    
    
    %Remove the baseline from the previous 200ms
    EEG = pop_rmbase( EEG, [-200 0]);
    
    %Modify the dataset in the EEGlab main window
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    
    %save new dataset with ps id
    n = subject_list{1,s};
    filepath = 'Filtered_Epoched/';
    pop_saveset(EEG, 'filename', n , 'filepath', filepath);
end

eeglab redraw
