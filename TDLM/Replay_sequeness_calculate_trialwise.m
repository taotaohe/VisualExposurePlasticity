clear;
clc;
close all;
rng('shuffle')

%% Spects
TF = [0,1,0,0;0,0,1,0;0,0,0,1;0,0,0,0]; % transition matrix
TR = TF';
maxLag = 150; % evaluate time lag up to 600ms
cTime = 0:4:maxLag*4; % the milliseconds of each cross-correlation time lag
[~, pInds] = uperms(1:4,24); % unique permutation
uniquePerms=pInds;
nShuf = size(uniquePerms,1);
nstates=4;
maxTrials = 96; % the maximal trial numbers per condition and per subject

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
condNames = {'Full', 'Start-only', 'End-only'};
n_conditions = numel(condNames);

% get indices of selected channels
channel_all = importdata(fullfile(project_path, 'data_v5/misc_data/channels_all.txt'));
channel_occipital = importdata(fullfile(project_path, 'data_v5/misc_data/occipital_channels.txt'));
[~,idx] = ismember(channel_occipital, channel_all);

% get optimal time points
optimal_time = readNPY(fullfile(project_path, 'data_v5/saved_source_data/optimal_time_idx_occipital_acc_matrix.npy'));

sf = cell(n_subjects,1);  sb = cell(n_subjects,1);
sf2 = cell(n_subjects,1);  sb2 = cell(n_subjects,1);

%% Core function
parfor isubj = 1:n_subjects    
    disp(['Working on Subject ', selected_subj{isubj}])

    %% load MEG data (n_trials x n_channels x n_times)
    modelTrain_stim = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_modelTrain_epochs_all_resample250_ica-epo.mat'])); % epochs_all, label
    modelTrain_ITI  = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_modelTrain_epochs_all_resample250_ica_ITI-epo.mat'])); % epochs_all, label
    
    modelTrain_ITI.label = zeros(length(modelTrain_ITI.label),1);
    modelTrain_ITI_data = mean(modelTrain_ITI.epochs_all, 3); % average ITI period over time
    
    %% make training data
    nSensors = size(idx,1);
    betas1 = nan(nSensors, nstates); intercepts = nan(1,nstates);
    for iC=1:4 % for each states
        trainingData = [modelTrain_stim.epochs_all(:,:,optimal_time(isubj,iC)); modelTrain_ITI_data];
        trainingData = trainingData(:,idx); % n_trials x n_channels; occipital channels
        % Standardize features by removing the mean and scaling by standard variance
        trainingData = [normalize(trainingData(1:size(modelTrain_stim.epochs_all,1),:),'scale');
                        normalize(trainingData(size(modelTrain_stim.epochs_all,1)+1:end,:),'scale')]; 

        trainingLabels = [modelTrain_stim.label'; modelTrain_ITI.label];
      
        % train classifiers on training data   
        [betas1(:,iC), fitInfo] = lassoglm(trainingData, trainingLabels==iC, 'binomial', 'Alpha', 1, 'Lambda', 0.006, 'Standardize', false);
        intercepts(iC) = fitInfo.Intercept;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    params = load(fullfile(project_path, ['data_v5/behavioral_data/' selected_subj{isubj} '/MainTask/params_PostTest_' selected_subj{isubj} '_R01.mat']));
    test_dir_idx = (params.p.Orient+90)/90;
    mainPost = load(fullfile(project_path, ['data_v5/' selected_subj{isubj} '/' selected_subj{isubj} '_mainPost_epochs_all_resample250_ica-epo.mat'])); % epochs_all, label
    
    sf{isubj} = nan(1, n_conditions, maxTrials, nShuf, maxLag+1);
    sb{isubj} = nan(1, n_conditions, maxTrials, nShuf, maxLag+1);
    sf2{isubj} = nan(1, n_conditions, maxTrials, nShuf, maxLag+1);
    sb2{isubj} = nan(1, n_conditions, maxTrials, nShuf, maxLag+1);
    
    % loop for 3 conditions
    for icond = 1:n_conditions        
        if icond == 1 % full sequeness
            cond_data  = mainPost.epochs_all(mainPost.label==test_dir_idx(1),idx,:);
        elseif icond == 2 % start-only
            cond_data  = mainPost.epochs_all(mainPost.label==test_dir_idx(1)+10,idx,:);
        elseif icond == 3 % end-only
            cond_data  = mainPost.epochs_all(mainPost.label==test_dir_idx(4)+10,idx,:);
        end

        for iTrial = 1:size(cond_data,1)
            X = normalize(squeeze(cond_data(iTrial,:,76:end))', 'scale');           

            % make predictions with trained models
            preds  = 1./(1+exp(-(X*betas1 + repmat(intercepts, [size(X,1) 1]))));

            % calculate sequenceness 
            for iShuf = 1:nShuf
                rp = uniquePerms(iShuf,:);  % use the 24 unique permutations
                T1 = TF(rp,rp); T2 = T1'; % backwards is transpose of forwards
                X = preds;

                nbins=maxLag+1;

               warning off
               dm=[toeplitz(X(:,1),[zeros(nbins,1)])];
               dm=dm(:,2:end);

               for kk=2:nstates
                   temp=toeplitz(X(:,kk),[zeros(nbins,1)]);
                   temp=temp(:,2:end);
                   dm=[dm temp]; 
               end

               warning on

               Y=X;       
               betas = nan(nstates*maxLag, nstates);

              %% GLM: state regression, with other lages
               bins=maxLag;

               for ilag=1:bins % first-level sequence analysis
                   temp_zinds = (1:bins:nstates*maxLag) + ilag - 1; 
                   temp = pinv([dm(:,temp_zinds) ones(length(dm(:,temp_zinds)),1)])*Y;
                   betas(temp_zinds,:)=temp(1:end-1,:);           
               end  

               betasnbins16=reshape(betas,[maxLag nstates^2]);
               bbb=pinv([T1(:) T2(:) squash(eye(nstates)) squash(ones(nstates))])*(betasnbins16'); % second-level sequence analysis

    %            sf{isubj}(1,icond,iTrial,iShuf,2:end) = zscore(bbb(1,:)); 
    %            sb{isubj}(1,icond,iTrial,iShuf,2:end) = zscore(bbb(2,:)); 
               sf{isubj}(1,icond,iTrial,iShuf,2:end) = bbb(1,:); 
               sb{isubj}(1,icond,iTrial,iShuf,2:end) = bbb(2,:); 

              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
              %% Cross-Correlation
              for iLag=1:maxLag
                  sf2{isubj}(1,icond,iTrial,iShuf,iLag+1) = sequenceness_Crosscorr(X, T1, [], iLag);
                  sb2{isubj}(1,icond,iTrial,iShuf,iLag+1) = sequenceness_Crosscorr(X, T2, [], iLag);
              end        
            end
        end
    end
end

sf_all = cell2mat(sf);
sb_all = cell2mat(sb);

sf2_all = cell2mat(sf2);
sb2_all = cell2mat(sb2);

%%% save the data
project_path = 'E:\Dropbox\Projects\featureReplay\data_v5';
save_path = fullfile(project_path,'/saved_source_data/sf_all_occipital_trialwise.mat');
save(save_path,'sf_all');
save_path = fullfile(project_path,'/saved_source_data/sb_all_occipital_trialwise.mat');
save(save_path,'sb_all');
save_path = fullfile(project_path,'/saved_source_data/sf2_all_occipital_trialwise.mat');
save(save_path,'sf2_all');
save_path = fullfile(project_path,'/saved_source_data/sb2_all_occipital_trialwise.mat');
save(save_path,'sb2_all');

% save all the possible sequeneces
save_path = fullfile(project_path,'/saved_source_data/unique_perms.mat');
save(save_path,'uniquePerms');
