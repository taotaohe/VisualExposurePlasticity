clear;
clc;
close all;

%% Preparation
% Set general parameters
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
condNames = {'Full', 'start-only', 'end-only'};
n_conditions = numel(condNames);

% get indices of selected channels
channel_all = importdata(fullfile(project_path, 'data_v5/misc_data/channels_all.txt'));
channel_occipital = importdata(fullfile(project_path, 'data_v5/misc_data/occipital_channels.txt'));
[~,idx] = ismember(channel_occipital, channel_all);

% get optimal time points
optimal_time = readNPY(fullfile(project_path, 'data_v5/saved_source_data/optimal_time_idx_occipital_acc_matrix.npy'));
sub_optimal_time = optimal_time+1;

%% Core function
proba = nan(n_subjects, n_conditions, 48, 775, 4); % n_subjs, n_conditions, n_trials(max), n_times(775), n_models(4)
for isubj = 1:n_subjects    
    disp(['Working on Subject ', selected_subj{isubj}])

    %% load ModelTrain data (n_trials x n_channels x n_times)
    modelTrain_stim = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_modelTrain_epochs_all_resample250_ica-epo.mat'])); % epochs_all, label
    modelTrain_ITI  = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_modelTrain_epochs_all_resample250_ica_ITI-epo.mat'])); % epochs_all, label
    
    modelTrain_ITI.label = zeros(length(modelTrain_ITI.label),1);
    modelTrain_ITI_data = mean(modelTrain_ITI.epochs_all, 3); % average ITI period over time
    
    %% make training data
    nSensors = size(idx,1); % nSensors = 306;
    betas = nan(nSensors, 4); intercepts = nan(1,4);
    for iC=1:4 % for each states (model)
        trainingData = [modelTrain_stim.epochs_all(:,:,sub_optimal_time(isubj,iC)); modelTrain_ITI_data];
        trainingData = trainingData(:,idx); % n_trials x n_channels; occipital channels
        % Standardize features by removing the mean and scaling to unit variance
        trainingData = [normalize(trainingData(1:size(modelTrain_stim.epochs_all,1),:),'scale');
                        normalize(trainingData(size(modelTrain_stim.epochs_all,1)+1:end,:),'scale')]; 
        trainingLabels = [modelTrain_stim.label'; modelTrain_ITI.label];
      
        % train classifiers on training data   
        [betas(:,iC), fitInfo] = lassoglm(trainingData, trainingLabels==iC, 'binomial', 'Alpha', 1, 'Lambda', 0.006, 'Standardize', false);
        intercepts(iC) = fitInfo.Intercept;
    end    
    
    %% apply training data to the main task data    
    params = load(fullfile(project_path, ['data_v5/behavioral_data/' selected_subj{isubj} '/MainTask/params_PostTest_' selected_subj{isubj} '_R01.mat']));
    test_dir_idx = (params.p.Orient+90)/90;
    mainPost = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_mainPost_epochs_all_resample250_ica-epo.mat'])); % epochs_all, label
    
    % loop for 4 conditions
    for icond = 1:n_conditions        
        if icond == 1 % full sequeness
            cond_data  = mainPost.epochs_all(mainPost.label==test_dir_idx(1),idx,:); % cond_data = mainPost.epochs_all(mainPost.label==test_dir_idx(1),2:307,:); whole brain sensors
        elseif icond == 2 % start-only
            cond_data  = mainPost.epochs_all(mainPost.label==test_dir_idx(1)+10,idx,:);
        elseif icond == 3 % end-only
            cond_data  = mainPost.epochs_all(mainPost.label==test_dir_idx(4)+10,idx,:);
        end

        for iTrial = 1:size(cond_data,1)
%             X = normalize(squeeze(cond_data(iTrial,:,76:end))', 'scale'); % only include time period after the onset of the first grating
            X = normalize(squeeze(cond_data(iTrial,:,:))', 'scale');

           %% make predictions with trained models
            proba(isubj,icond,iTrial,:,:)  = 1./(1+exp(-(X*betas(:,test_dir_idx) + repmat(intercepts(:,test_dir_idx), [size(X,1) 1]))));    
        end
    end
end

%% save the data
save_path = fullfile(project_path,'/data_v5/saved_source_data/proba_mainTask.mat');
save(save_path,'proba');
