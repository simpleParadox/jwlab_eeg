addpath '~/projects/def-jwerker/kjslakov/eeglab2019_0'
eeglab

% set up the script parameters
FILEPATH = '~/projects/def-jwerker/kjslakov/data/';
FILEPATH_OUT = '~/projects/def-jwerker/kjslakov/data/cleaned/';
SUBJECTS = {'107', '111', '112', '115', '116', '904', '905', '906', '908', '909', '910', '912', '913', '914', '916'};

ML_EVENTS = { 'Wait' };
PIC_EVENTS = { 'Pict' };
BOUNDARY_EVENTS = { 'boundary' };

SAVE_INTERMEDIATE = false;

for curr_subject = SUBJECTS
    % load each subjects data for cleaning
    curr_EEG = pop_loadset('filename', char(append(curr_subject, '.set')), 'filepath', FILEPATH);
    
    % TODO
    curr_EEG = clean_bad_channels(curr_EEG);
    
    % First, remove data around "boundary" events, as the data there is
    % corrupted
    curr_EEG = pop_rmdat(curr_EEG, BOUNDARY_EVENTS, [-0.5, 0.1], 1);
    
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
    
    % do avg referencing -- TODO : look into if you're sure this is working
    ml_eeg = pop_reref(ml_eeg, []);
    pic_eeg = pop_reref(pic_eeg, []);
     
    pop_saveset(ml_eeg, 'filename', char(append(curr_subject, '_cleaned_ml.set')), 'filepath', FILEPATH_OUT);
    pop_saveset(pic_eeg, 'filename', char(append(curr_subject, '_cleaned_pic.set')), 'filepath', FILEPATH_OUT);
    
    pop_export(ml_eeg, char(append(FILEPATH_OUT, curr_subject, '_cleaned_ml.csv')), 'transpose', 'on', 'separator', ',');
    pop_export(pic_eeg, char(append(FILEPATH_OUT, curr_subject, '_cleaned_pic.csv')), 'transpose', 'on', 'separator', ',');
end







