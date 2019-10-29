%This code selects a random subset of animate and inanimate
%trials from each particpant to create 10% of each type for test set
%the remainder is left for a training set

%for many participants
subject_list = {'105' '107' '106' '904' '905' '906'};
%'105' '107' '106' '904' '905' '906'
num_subjects = length(subject_list);
Indiv = [];
accuracy_trainingAll_ForEachPS_ByIteration =[];
IndividualTestMatrices = [];
IndividualLabelMatrices = [];
individualClassificaiton_allTrain = [];
IndividualTestMatrices_byIteration = [];
IndividualLabelMatrices_byIteration = [];

for ind = 1:10 % this will take 10random samples and 
    %tell us the summary of each
    joint_EEG_Train_Data = [];
    joint_EEG_Test_Data = [];
    joint_EEG_Train_Label = [];
    joint_EEG_Test_Label =[];
    


    for s = 1:num_subjects
        indivInd = s;
        animates = [];
        inanimates = [];
        masterMatrix = [];
        classification = [];

        %filepath = 'Filtered_Epoched/';
        filepath = 'Bad_Trials_Removed/';
        EEG = pop_loadset('filename', [subject_list{s} '.set'], 'filepath', filepath);
        EEG.setname = [subject_list{s} '.set'];


        %create trial code, entire data file
        num_channels = size(EEG.data,1);
        num_samples = size(EEG.data,2);
        num_trials = size(EEG.data,3);

        newEEG = zeros(num_trials, num_channels * num_samples);

        for i = 1:num_trials
            for j =1:num_channels
                newEEG(i, (j-1)*num_samples+1:j*num_samples) = EEG.data(j,:,i);
            end
        end 

        %classification matrix
        class(:,1) = {EEG.epoch(:).eventmffkey_cel};

        for k = 1:num_trials
            valuecell = class{k,1};
            if ischar(valuecell) 
                value = str2double(valuecell);
            else   
                cellarray = valuecell{1,1};
                value = str2double(cellarray);
            end 
            classification = [classification; value];
        end


        masterMatrix = [classification newEEG]; %puts our class codes into a single

        for k = 1:num_trials
            if masterMatrix(k,1) == 1 %remember matlab does row then colum
                animates = [animates; masterMatrix(k,:)];
            elseif masterMatrix(k,1) == 2 
                animates = [animates; masterMatrix(k,:)];
            elseif masterMatrix(k,1) == 3 
                inanimates = [inanimates; masterMatrix(k,:)];
            elseif masterMatrix(k,1) == 4
                inanimates = [inanimates; masterMatrix(k,:)];
            end
        end

    %Lump animates and inanimates into codes 1 and 2
    animates(:, 1) = 1; 
    inanimates(:, 1) = 2; 


    anim_size = size(animates,1);
    tenPerAn = ceil(anim_size*.1);
    inanim_size = size(inanimates,1);
    tenPerInan = ceil(inanim_size*.1);

    %get subsamples
    %gives the training set and vector of indexs pulled
    [EEG_Test_Anim, index_anim] = datasample(animates,tenPerAn,'Replace',false);
    EEG_Train_Anim = animates;
    EEG_Train_Anim(index_anim,:) =[]; %removes those rows
    
    [EEG_Test_Inanim, index_inanim] = datasample(inanimates,tenPerInan,'Replace',false);
    EEG_Train_Inanim = inanimates;
    EEG_Train_Inanim(index_inanim,:) =[];

    EEG_Train_Data = [EEG_Train_Anim; EEG_Train_Inanim];
    EEG_Test_Data = [EEG_Test_Anim; EEG_Test_Inanim];
    EEG_Train_Label = EEG_Train_Data(:,1);
    EEG_Test_Label  = EEG_Test_Data(:,1);

    EEG_Train_Data(:, 1) = []; %remove first column
    EEG_Test_Data(:, 1) = []; %remove first column

    %build the joint databases
    joint_EEG_Train_Data = [joint_EEG_Train_Data; EEG_Train_Data];
    joint_EEG_Test_Data = [joint_EEG_Test_Data; EEG_Test_Data];
    joint_EEG_Train_Label = [joint_EEG_Train_Label; EEG_Train_Label];
    joint_EEG_Test_Label = [joint_EEG_Test_Label; EEG_Test_Label];


    %classification using inidividual matrices
    %this gives the instructions: doc fitcecoc 
    Mdl = fitcecoc(EEG_Train_Data,EEG_Train_Label); 
    %gives a matrix of the predicted classes
    p = predict(Mdl,EEG_Test_Data);
    %this will give you the classification table
    ConMat = confusionmat(EEG_Test_Label, p);
    accuracyIndiv = mean(p == EEG_Test_Label);
    IndivTemp = [];
    s = subject_list{s};
    IndivTemp = [ind str2num(s) accuracyIndiv];
    %avg for each participant for each run
    Indiv = [Indiv; IndivTemp]; 
    
    %store each indiviudal test data and label
    IndividualTestMatrices{indivInd} = EEG_Test_Data;
    IndividualLabelMatrices{indivInd} = EEG_Test_Label;
    
    
    clear accuracyIndiv;
    clear IndivTemp;
    clear class;

    end 

    %classification
    %this gives the instructions: doc fitcecoc 
    Mdl = fitcecoc(joint_EEG_Train_Data,joint_EEG_Train_Label); 

    %gives a matrix of the predicted calsses
    p = predict(Mdl,joint_EEG_Test_Data);
    %this will give you the classification table
    ConMat = confusionmat(joint_EEG_Test_Label, p);
    accuracy = mean(p == joint_EEG_Test_Label);
    accuracyTemp = [ind accuracy];
    accuracy_trainingAll_ForEachPS_ByIteration = [accuracy_trainingAll_ForEachPS_ByIteration; accuracyTemp];
    clear accuracyTemp;
  
    IndividualTestMatrices_byIteration = [IndividualTestMatrices_byIteration; IndividualTestMatrices];
    IndividualLabelMatrices_byIteration = [IndividualLabelMatrices_byIteration; IndividualLabelMatrices];
end 


accuracy_trainingAll_ForEachPS_ByIteration %avg all ps by iteration
accuracyTotal_trainingAll_testAll = mean(accuracy_trainingAll_ForEachPS_ByIteration(:,2)); %all ps
accuracyTotal_trainingAll_testAll

%for all iterations, accuracy means for each subject
%using individual for training
for i = 1:num_subjects
    individualClassificaiton_indivTrain(i,1) = str2num(subject_list{i});
    individualClassificaiton_indivTrain(i,2) = mean(Indiv(i:num_subjects:length(Indiv),3));
end

%%
%classification using joint training matrix
%%something really weird is happening here! 
individualClassificaiton_allTrain = [];
accuracyTempID = [];
accuracyAvg = [];
Mdl = fitcecoc(joint_EEG_Train_Data,joint_EEG_Train_Label); 
for l = 1:num_subjects
    ps = str2num(subject_list{l});
    accuracyRun = [];
    for y = 1:size(IndividualTestMatrices_byIteration)
        p = predict(Mdl,IndividualTestMatrices_byIteration{y,l});
        accuracy = mean(p == IndividualLabelMatrices_byIteration{y,l});
        accuracyTemp = [ps accuracy];
        accuracyRun = [accuracyRun; accuracyTemp];
    end
    accuracyAvg = [mean(accuracyRun)];
    individualClassificaiton_allTrain = [individualClassificaiton_allTrain; accuracyAvg];
end 

% 
% %
% Indiv %complete chart, using individual for trianing
% individualClassificaiton_indivTrain %ps results when trained on own data
% individualClassificaiton_allTrain %ps results when trained on all data
% 
%accuracy_trainingAll_ForEachPS_ByIteration %avg all ps by iteration
%accuracyTotal_trainingAll_testAll % single value average