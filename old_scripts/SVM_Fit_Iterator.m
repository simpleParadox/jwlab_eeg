
%Different fit functions for SVM code:
%run after Matrix construction code
joint_EEG_Train_Data 
    joint_EEG_Test_Data
    joint_EEG_Train_Label 
    joint_EEG_Test_Label 
    
%Define the values of c that we want to use
C = [1E-7,1E-5,1E-3,.1,1,5,10];
accuracy = zeros(1,length(C));
mean_train_data = mean(joint_EEG_Train_Data);

% Here we used PCA to reduce this down
zero_mean_train_data = joint_EEG_Train_Data - mean_train_data;
zero_mean_test_data = joint_EEG_Test_Data - mean_train_data;

coeff = pca(zero_mean_train_data);

z_train = joint_EEG_Train_Data*coeff;
z_test = joint_EEG_Test_Data*coeff;

%change from train_data
X_train = z_train;
X_test = z_test;

%iterate through all the values
for i = 1:length(C)
    i
    %take the i-th value of C
    current_c = C(i);
    %create the SVM template
    %value of c
    %allow it to terminate
    t = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', ...
        current_c, 'IterationLimit', 1000);
    model = fitcecoc(X_train,joint_EEG_Train_Label ,'Learners', t);
    p = predict(model, X_test);
    ConfMat = confusionmat(joint_EEG_Test_Label , p);
    accuracyCFit(i) = mean(p == joint_EEG_Test_Label);
    
    accuracyCFit
    
    
end