clear;
clc;
close all;

%% Preparation
% Set general parameters
% addpath('helper_functions');
[status, result] = system('who');  if status machine = 'pc'; elseif ~status machine = 'linux'; end % determine the operation system 
if isequal(machine,'linux')
    project_path = '/home/user/Projects/featureReplay'; % Change your path
elseif isequal(machine,'pc')
    project_path = 'D:\Dropbox\Projects\featureReplay';
end

SUBJECTS = {'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', ...
            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', ...
            'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', ...
            'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'};

selection = 1:18; 
selected_subj = SUBJECTS(selection);
n_subjects = numel(selected_subj);

% get indices of selected channels
channel_all = importdata(fullfile(project_path, 'data_v5/misc_data/channels_all.txt'));
channel_occipital = importdata(fullfile(project_path, 'data_v5/misc_data/occipital_channels.txt'));
[~,idx] = ismember(channel_occipital, channel_all);

% get optimal time points
optimal_time = readNPY(fullfile(project_path, 'data_v5/saved_source_data/optimal_time_idx_occipital_acc_matrix.npy'));
sub_optimal_time = optimal_time+1;

%% Core function
preds = nan(n_subjects,4,4,325); % n_subject, n_true_label, n_predict, n_times
betas_save = nan(n_subjects,4,length(idx));
for isubj = 1:n_subjects    
    disp(['Working on Subject ', selected_subj{isubj}])

    %% load ModelTrain data (n_trials x n_channels x n_times)
    modelTrain_stim = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_modelTrain_epochs_all_resample250_ica-epo.mat'])); % epochs_all, label
    modelTrain_ITI  = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_modelTrain_epochs_all_resample250_ica_ITI-epo.mat'])); % epochs_all, label
    
    modelTrain_ITI.label = zeros(length(modelTrain_ITI.label),1);
    modelTrain_ITI_data = mean(modelTrain_ITI.epochs_all, 3); % average ITI period over time
    
    %% make training data
    for imodel=1:4 % four models (4 optimal times)        
        trainingData = [modelTrain_stim.epochs_all(:,:,sub_optimal_time(isubj,imodel)); modelTrain_ITI_data];
        trainingData = trainingData(:,idx); % n_trials x n_channels; occipital channels
        % Standardize features by removing the mean and scaling by standard variance
        trainingData = [normalize(trainingData(1:size(modelTrain_stim.epochs_all,1),:),'scale');
                        normalize(trainingData(size(modelTrain_stim.epochs_all,1)+1:end,:),'scale')]; 
        trainingLabels = [modelTrain_stim.label'; modelTrain_ITI.label];
      
        % loop over stimulus epochs only, do not include ITI epochs
        proba_y = nan(length(modelTrain_stim.label), size(modelTrain_stim.epochs_all,3)); % n_trials, n_times
        betas_tmp = nan(length(modelTrain_stim.label), length(idx)); % n_trials, n_features
        for iTrial = 1:length(nonzeros(trainingLabels))
                    
            tData = trainingData; tData(iTrial,:) = []; % leave one trial out
            tLabel = trainingLabels; tLabel(iTrial) = []; % leave the corresponding trial label out

            % train classifiers on training data   
            [betas, fitInfo] = lassoglm(tData, tLabel==imodel, 'binomial', 'Alpha', 1, 'Lambda', 0.006, 'Standardize', false);
            intercepts = fitInfo.Intercept;  
            betas_tmp(iTrial,:) = betas'; % temporally save betas of each trial, for plotting correlation coefficient

            % test for each time point on the left one trial
            for iTime = 1:size(modelTrain_stim.epochs_all,3) %time points
                testData = normalize(modelTrain_stim.epochs_all(:,idx,iTime), 'scale');
                testData = testData(iTrial,:);
                % make predictions with trained models
                proba_y(iTrial, iTime) = 1./(1+exp(-(testData*betas + repmat(intercepts, [size(testData,1) 1]))));
            end
        end
        betas_save(isubj,imodel,:) = squeeze(mean(betas_tmp, 1)); % save for plotting correlation coefficient

        for ilabel = 1:4 % average over trials with the same label
            preds(isubj,ilabel,imodel,:) = squeeze(mean(proba_y(modelTrain_stim.label==ilabel,:), 1)); % n_subj, n_true_labels, n_pred_models, n_times
        end
    end
end

%% save the data
save_path = fullfile(project_path,'/data_v5/saved_source_data/preds_modelTrain.mat');
save(save_path,'preds');

save_path = fullfile(project_path,'/data_v5/saved_source_data/betas_modelTrain.mat');
save(save_path,'betas_save');

