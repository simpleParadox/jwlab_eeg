%% Sanity check
eeglab

subject_list = {'105' '106' '107' '904' '905' '906'};
num_subjects = length(subject_list);

word_list = {'baby' 'bear' 'bird' 'bunny' 'cat' 'dog' 'duck' 'mommy' 'banana' 'bottle' 'cookie' 'cracker' 'cup' 'juice' 'milk' 'spoon'};

words{1} = {'fork' 'cup' 'spoon' 'turkey' 'cheese' 'cup' 'bottle' 'dog' 'bear' 'baby' 'cookie' 'egg' 'banana' 'bear'...
    'cracker' 'cracker' 'cookie' 'bottle' 'mommy' 'milk' 'bottle' 'milk' 'horse' 'orange' 'fish' 'egg' 'cat' 'squirrel'...
    'bird' 'juice' 'spoon' 'bunny' 'bottle' 'fork' 'cat' 'bunny' 'juice' 'duck' 'cup' 'bird' 'bird'};
    
words{2} = {'cup' 'horse' 'fork' 'duck' 'cracker' 'cat' 'orange' 'bear' 'duck' 'cat' 'juice' 'turkey' 'turkey' 'cheese' ...
    '-' 'egg' 'bottle' 'bear' 'bottle' 'cookie' 'apple' 'fish' 'cookie' 'egg' 'fork' 'bird' 'orange' 'baby' 'dog' 'mommy' 'milk'};
    
words{3} = {'bear' 'spoon' 'bird' 'cracker' 'bunny' 'bottle' 'duck' 'bottle' 'dog' 'milk' 'bunny' 'cup' 'bird' 'milk' 'cat' 'cookie' 'mommy'...
    'cup' 'duck' 'spoon' 'cat' 'juice' 'baby' 'cracker' 'baby' 'cookie' 'mommy' 'juice' 'bear' 'banana' 'dog' 'milk' 'duck' 'cracker'...
    'juice' 'bear''cookie' 'dog' 'banana' 'bird' 'dog' '-' '-' 'juice' 'baby' 'cup' 'cat' 'duck' 'banana' 'bear' 'cracker' 'bird'...
    'bottle' 'mommy' 'spoon' 'baby' 'cookie' 'bird' 'cookie' 'cat' 'bird' 'banana' 'duck' 'milk' 'mommy' 'spoon' 'mommy' 'cracker'...
    'baby' 'bottle' 'bunny' 'juice' 'bunny' 'cup' 'baby' 'bottle' 'juice' 'dog' 'spoon' 'cracker' 'duck' 'juice' 'bunny' 'spoon' 'bear' 'bottle' 'mommy'};
    
words{4} = {'bear' 'cookie' 'mommy' 'banana' 'duck' 'juice' 'dog' 'banana' 'bird' 'cracker' 'spoon' 'duck' 'cup' 'cat' 'cookie' 'bunny'...
    'spoon' 'duck' 'cup' 'cat' 'milk' 'bear' 'bottle' 'mommy' 'milk' 'cracker' 'cat' 'juice' 'dog' 'cookie' 'milk' 'cookie' 'duck'...
    'spoon' 'mommy' 'spoon' 'duck' 'cracker' 'bear' 'bottle' 'baby' 'cracker' 'mommy' 'cup' '-' 'banana' 'bird' 'banana' '-' 'juice'...
    'cat' 'duck' 'banana' 'bear' 'cookie' 'bird' 'juice' 'dog' 'bear' '-' 'baby' '-' 'cat' 'cracker' 'bunny' 'juice' 'mommy' 'milk'...
    'mommy' 'bird' 'cup' 'dog' 'banana' 'duck' 'spoon' 'milk' 'cracker' 'banana' 'cookie' 'duck' 'cup' 'bunny' 'juice' 'bunny' 'bottle' 'juice' 'spoon'};
    
words{5} = {'baby' 'banana' 'bear' 'banana' 'mommy' 'cracker' 'bear' 'cup' 'duck' 'cookie' 'bunny' 'cup' 'dog' 'cracker' 'dog' 'juice' 'cat'...
    'spoon' 'mommy' 'juice' 'bunny' 'bottle' 'bird' 'bottle' 'cat' 'milk' 'bird' 'spoon' 'duck' 'cookie' 'baby' 'milk' 'mommy' 'banana'...
    'bear' 'juice' 'cat' 'baby' 'duck' 'bottle' 'bird' 'cracker' 'bunny' 'cookie' 'baby' 'milk' 'cat' 'spoon' 'bunny' 'juice' 'dog' 'spoon'...
    'bird' 'bottle' 'duck' 'cup' 'mommy' 'cup' 'dog' 'cookie' 'bear' 'cracker' 'mommy' 'cup' 'duck' 'bird' 'milk' 'baby' '-' 'dog'...
    'juice' 'bear' 'cup' 'duck' 'milk' 'bird' 'dog' 'bottle'};
    
words{6} = {'cat' 'cat' 'bunny' 'banana' 'bear' 'bird' 'bottle' 'dog' 'mommy' 'juice' 'duck' 'bottle' 'baby' 'cup' 'milk' 'cracker' 'juice'...
    'bird' 'mommy' 'milk' 'baby' 'duck' 'juice' 'baby' 'banana' 'duck' 'cup' 'bottle' 'cup' 'duck'};

filepath = '/Volumes/JWLAB/Members_Current/Jenn/EEG study/Matlab codes/Bad_Trials_Removed/';

for s = 1:num_subjects
    %Load the dataset according to the vector containing the subjects tags
    EEG = pop_loadset('filename', [subject_list{s} '.set'], 'filepath', filepath);
    EEG.setname = [subject_list{s} '_EEG'];
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
end
eeglab redraw

%% Individual classification
for s = 1:num_subjects
    clear newEEG class classification train_b train_a

    num_channels = size(ALLEEG(s).data,1);
    num_samples = size(ALLEEG(s).data,2);
    num_trials = size(ALLEEG(s).data,3);

    bEEG = zeros(num_trials, num_channels * 200);
    aEEG = zeros(num_trials, num_channels * 200);

    for i = 1:num_trials
        for j = 1:num_channels
            %Extract the 200ms previous to the stimuli
            bEEG(i,(j-1)*200+1:j*200) = ALLEEG(s).data(j,1:200,i);

            %Sliding window of 200 samples with a step size of 50
            %Extract 200ms intervals after the stimuli
            stp = 50;
            for k = 1:17
                aEEG(i,(j-1)*200+1:j*200,k) = ALLEEG(s).data(j,(k-1)*stp+201:(k-1)*stp+400,i);
            end
        end
    end

    for l = 1:size(aEEG,3)
        accuracy = zeros(10,1);
        iterations = 10;

        for j = 1:iterations
            train_b = bEEG;
            train_a = aEEG;
            
            clear test_b test_a idx_b idx_a test_data train_data test_label train_label Mdl p
            [test_b, idx_b(:,1)] = datasample(train_b,ceil(size(train_b,1)/10),1,'Replace',false);
            [test_a, idx_a(:,1)] = datasample(train_a,ceil(size(train_a,1)/10),1,'Replace',false);

            train_b(idx_b,:) = [];
            train_a(idx_a,:,:) = [];

            test_data = [test_b;test_a(:,:,l)];
            train_data = [train_b;train_a(:,:,l)];
            test_label = [ones(size(test_b,1),1);2*ones(size(test_a,1),1)];
            train_label = [ones(size(train_b,1),1);2*ones(size(train_a,1),1)];

            Mdl = fitcecoc(train_data,train_label);
            p = predict(Mdl,test_data);

            ConMatSlid(:,:,j,s) = confusionmat(test_label, p);

            accuracy(j) = mean(p == test_label);
            
            progress = (s-1)*size(aEEG,3)*iterations+(l-1)*iterations+j;
            completition = num_subjects*size(aEEG,3)*iterations;
            disp(['Individual classification: ' num2str(100*progress/completition) '% complete']);
        end

        acc(l,s) = mean(accuracy);

    end

end

%% Group classification

jointEEG = cell(length(word_list),num_subjects);
num_words = length(word_list);

% Trial average according to word
for w = 1:num_words
    for s = 1:num_subjects
        word_loc = find(contains(words{s},word_list{w}));
        jointEEG{w,s} = ALLEEG(s).data(:,:,word_loc);
        if s>1
            jointEEG{w,1} = cat(3,jointEEG{w,1},jointEEG{w,s});
        end
    end
    jointEEG{w,1} = mean(jointEEG{w,1},3);
end

groupEEG = zeros(num_channels,num_samples,num_words);

for w = 1:num_words
    groupEEG(:,:,w) = [jointEEG{w,1}];
end

bEEG = zeros(num_words, num_channels * 200);
aEEG = zeros(num_words, num_channels * 200);

for i = 1:num_words
    for j = 1:num_channels
        %Extract the 200ms previous to the stimuli
        bEEG(i,(j-1)*200+1:j*200) = groupEEG(j,1:200,i);

        %Sliding window of 200 samples with a step size of 50
        %Extract 200ms intervals after the stimuli
        stp = 50;
        for k = 1:17
            aEEG(i,(j-1)*200+1:j*200,k) = groupEEG(j,(k-1)*stp+201:(k-1)*stp+400,i);
        end
    end
end

for l = 1:size(aEEG,3)
    accuracy = zeros(10,1);
    iterations = 10;

    for j = 1:iterations
        train_b = bEEG;
        train_a = aEEG;

        clear test_b test_a idx_b idx_a test_data train_data test_label train_label Mdl p
        [test_b, idx_b(:,1)] = datasample(train_b,ceil(size(train_b,1)/10),1,'Replace',false);
        [test_a, idx_a(:,1)] = datasample(train_a,ceil(size(train_a,1)/10),1,'Replace',false);

        train_b(idx_b,:) = [];
        train_a(idx_a,:,:) = [];

        test_data = [test_b;test_a(:,:,l)];
        train_data = [train_b;train_a(:,:,l)];
        test_label = [ones(size(test_b,1),1);2*ones(size(test_a,1),1)];
        train_label = [ones(size(train_b,1),1);2*ones(size(train_a,1),1)];

        Mdl = fitcecoc(train_data,train_label);
        p = predict(Mdl,test_data);

        ConMatSlid(:,:,j,s) = confusionmat(test_label, p);

        accuracy(j) = mean(p == test_label);

        progress = (l-1)*iterations+j;
        completition = size(aEEG,3)*iterations;
        disp(['Group classification: ' num2str(100*progress/completition) '% complete']);
    end

    acc(l,num_subjects+1) = mean(accuracy);

end

%% Ploting

figure(3)
start_time = 0:50:800;
for s = 1:num_subjects
    subplot(ceil(num_subjects)/2,2,s)
    scatter(start_time,acc(:,s),'filled');
    
    grid on
    ylabel('Accuracy')
    xlabel('Starting time of the sliding window (ms)')
    title(['Subject ' subject_list{s}])
    ylim([0 1])
end

figure(4)
scatter(start_time,acc(:,num_subjects+1),'filled');

grid on
title('Average across words')
ylim([0 1])
ylabel('Accuracy')
xlabel('Starting time of the sliding window (ms)')