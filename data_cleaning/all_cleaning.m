% set up the script parameters
FILEPATH = '../data/';
SUBJECTS = {'999'};

ML_EVENTS = { 'Wait' };
PIC_EVENTS = { 'Pict' };
BOUNDARY_EVENTS = { 'boundary' };

SAVE_INTERMEDIATE = true;

% we'll gather all cleaned data into this array
subject_eegreadings = {};

for curr_subject = SUBJECTS
    % load each subjects data for cleaning
    curr_EEG = pop_loadset('filename', char(append(curr_subject, '.set')), 'filepath', FILEPATH);
    
    % TODO
    curr_EEG = clean_bad_channels(curr_EEG);
    
    % First, remove data around "boundary" events, as the data there is
    % corrupted
    curr_EEG = pop_rmdat(EEG, BOUNDARY_EVENTS, [-0.5, 0.1], 1);
    
    if SAVE_INTERMEDIATE
        pop_saveset(curr_EEG, 'filename', 'no_boundaries.set', 'filepath', FILEPATH);
    end
    
    % Now break into two sets, ML and picture, and apply appropriate filter
    ml_eeg = pop_eegfiltnew(curr_EEG, 1, 50);
    pic_eeg = pop_eegfiltnew(curr_EEG, 3, 30);
    
    if SAVE_INTERMEDIATE
        pop_saveset(ml_eeg, 'filename', 'ml_filtered.set', 'filepath', FILEPATH);
        pop_saveset(pic_eeg, 'filename', 'pic_filtered.set', 'filepath', FILEPATH);
    end
    
    ml_eeg = pop_epoch(ml_eeg, ML_EVENTS, [-0.2, 1]);
    pic_eeg = pop_epoch(pic_eeg, PIC_EVENTS, [-0.2, 1]);
    
    
    if SAVE_INTERMEDIATE
        pop_saveset(ml_eeg, 'filename', 'ml_epoched.set', 'filepath', FILEPATH);
        pop_saveset(pic_eeg, 'filename', 'pic_epoched.set', 'filepath', FILEPATH);
    end
    
    % do baseline removal from 200ms before world/picture onset
    ml_eeg = pop_rmbase(ml_eeg, [-200, 0]);
    pic_eeg = pop_rmbase(pic_eeg, [-200, 0]);
    
    % do avg referencing
    ml_eeg = pop_reref(ml_eeg, []);
    pic_eeg = pop_reref(pic_eeg, []);
    
    % remove all data other than the important points
    pic_eeg = pop_rmdat(pic_eeg, PIC_EVENTS, [-0.2, 1], 0);
    ml_eeg = pop_rmdat(ml_eeg, ML_EVENTS, [-0.2, 1], 0);
    
    ml_filename = char(append(curr_subject, '_cleaned_ml.set'));
    pic_filename = char(append(curr_subject, '_cleaned_pic.set'));    
    pop_saveset(ml_eeg, 'filename', ml_filename, 'filepath', FILEPATH);
    pop_saveset(pic_eeg, 'filename', pic_filename, 'filepath', FILEPATH);
end
