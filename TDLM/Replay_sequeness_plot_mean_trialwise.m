clear;
clc;
close all;

maxLag = 150; % evaluate time lag up to 600ms
cTime = 0:4:maxLag*4; % the milliseconds of each cross-correlation time lag

%% Preparation
SUBJECTS = {'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', ...
            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', ...
            'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', ...
            'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'};

selection = 1:18;
selected_subj = SUBJECTS(selection);
actual_subj = {'S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S22','S23','S24','S25'};
[~,subSubj] = ismember(selected_subj, actual_subj);
n_subjects = numel(selected_subj);
condNames = {'Full', 'Start-only', 'End-only'};
n_conditions = numel(condNames);

project_path = 'D:\Dropbox\Projects\featureReplay';
load(fullfile(project_path,'/data_v5/saved_source_data/sf_all_occipital_trialwise.mat'));
load(fullfile(project_path,'/data_v5/saved_source_data/sb_all_occipital_trialwise.mat'));
load(fullfile(project_path,'/data_v5/saved_source_data/sf2_all_occipital_trialwise.mat'));
load(fullfile(project_path,'/data_v5/saved_source_data/sb2_all_occipital_trialwise.mat'));

for icond = 1:n_conditions
    sf = squeeze(nanmean(sf_all(subSubj,icond,:,:,:),3)); %n_subjects x n_shuffle x n_times (mean across trials)
    sb = squeeze(nanmean(sb_all(subSubj,icond,:,:,:),3));
    sf2 = squeeze(nanmean(sf2_all(subSubj,icond,:,:,:),3));
    sb2 = squeeze(nanmean(sb2_all(subSubj,icond,:,:,:),3));  
    
    % calculate data for displaying in the paper
    npThresh = squeeze(max(abs(mean(sf(:,2:end,2:end)-sb(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);  
    dtp = squeeze(sf(:,1,:)-sb(:,1,:));
    replay_range{icond} = find(abs(mean(dtp))>npThreshAll) * 4;
    [m, i] = max(abs(mean(dtp)));
    replay_max{icond} = i*4;
    replay_max_mean{icond} = mean(dtp(:,i));
    replay_max_sem{icond} = std(dtp(:,i))/18;
    
    % start plotting from here
    fig = figure('Name',sprintf('%s condition - Mean',condNames{icond}));
    fig.Units = 'inches';
    fig.PaperUnits = 'inches';
    fig.PaperSize = [20, 10];
    fig.Position = [.1 .1 20 10];
    fig.PaperPositionMode = 'auto';
    
    %% GLM (fwd-bkw)
    subplot(2,3,1)
    npThresh = squeeze(max(abs(mean(sf(:,2:end,2:end)-sb(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);  
    dtp = squeeze(sf(:,1,:)-sb(:,1,:));
    shadedErrorBar(cTime, mean(dtp), std(dtp)/sqrt(n_subjects), {'-','color',[.25 .6 .9],'linewidth',1.5}, 0.5), hold on,
    plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
    title('GLM: fwd-bkw'), xlabel('lag (ms)'), ylabel('fwd minus bkw sequenceness')

    set(gca, 'XTick',(0:100:600), 'YTick',(-.03:.03:.03), 'YLim',[-0.03 0.03], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background
    
    %% GLM (fwd)
    subplot(2,3,2)
    npThresh = squeeze(max(abs(mean(sf(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);  
    dtp = squeeze(sf(:,1,:));
    shadedErrorBar(cTime, mean(dtp), std(dtp)/sqrt(n_subjects), {'-','color',[.25 .6 .9],'linewidth',1.5}, 0.5), hold on,
    plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
    title('GLM: fwd'), xlabel('lag (ms)'), ylabel('fwd sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.04:.04:.04), 'YLim',[-0.04 0.04], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% GLM (bkw)
    subplot(2,3,3)
    npThresh = squeeze(max(abs(mean(sb(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);  
    dtp = squeeze(sb(:,1,:));
    shadedErrorBar(cTime, mean(dtp), std(dtp)/sqrt(n_subjects), {'-','color',[.25 .6 .9],'linewidth',1.5}, 0.5), hold on,
    plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
    title('GLM: bkw'), xlabel('lag (ms)'), ylabel('bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.04:.04:.04), 'YLim',[-0.04 0.04], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% Cross-Correlation (fwd-bkw)
    subplot(2,3,4)
    npThresh = squeeze(max(abs(mean(sf2(:,2:end,2:end)-sb2(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);
    dtp = squeeze(sf2(:,1,:)-sb2(:,1,:));
    shadedErrorBar(cTime, mean(dtp), std(dtp)/sqrt(n_subjects), {'-','color',[.25 .6 .9],'linewidth',1.5}, 0.5), hold on,
    plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
    title('Correlation: fwd-bkw'), xlabel('lag (ms)'), ylabel('fwd minus bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.04:.04:.04), 'YLim',[-0.04 0.04], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% Cross-Correlation (fwd)
    subplot(2,3,5)
    npThresh = squeeze(max(abs(mean(sf2(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);
    dtp = squeeze(sf2(:,1,:));
    shadedErrorBar(cTime, mean(dtp), std(dtp)/sqrt(n_subjects), {'-','color',[.25 .6 .9],'linewidth',1.5}, 0.5), hold on,
    plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
    title('Correlation: fwd'), xlabel('lag (ms)'), ylabel('fwd sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.04:.04:.04), 'YLim',[-0.04 0.04], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% Cross-Correlation (bkw)
    subplot(2,3,6)
    npThresh = squeeze(max(abs(mean(sb2(:,2:end,2:end),1)),[],3));
    npThreshAll = prctile(npThresh, 95);
    dtp = squeeze(sb2(:,1,:));
    shadedErrorBar(cTime, mean(dtp), std(dtp)/sqrt(n_subjects), {'-','color',[.25 .6 .9],'linewidth',1.5}, 0.5), hold on,
    plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
    title('Correlation: bkw'), xlabel('lag (ms)'), ylabel('bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.04:.04:.04), 'YLim',[-0.04 0.04], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background
        

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig = figure('Name',sprintf('%s condition - Single subject',condNames{icond}));
    fig.Units = 'inches';
    fig.PaperUnits = 'inches';
    fig.PaperSize = [20, 10];
    fig.Position = [.1 .1 20 10];
    fig.PaperPositionMode = 'auto';

    %% GLM (fwd-bkw), single subject
    subplot(2,3,1)
    dtp = squeeze(sf(:,1,:)-sb(:,1,:));
    for isubj = 1:size(dtp,1)
        plot(cTime, dtp(isubj,:), 'LineWidth',1);
        hold on
    end
    title('GLM: fwd-bkw (single subject)'), xlabel('lag (ms)'), ylabel('fwd minus bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.15:.05:.15), 'YLim',[-0.15 0.15], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% GLM (fwd), single subject
    subplot(2,3,2)
    dtp = squeeze(sf(:,1,:));
    for isubj = 1:size(dtp,1)
        plot(cTime, dtp(isubj,:), 'LineWidth',1);
        hold on
    end
    title('GLM: fwd (single subject)'), xlabel('lag (ms)'), ylabel('fwd sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.15:.05:.15), 'YLim',[-0.15 0.15], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% GLM (fwd), single subject
    subplot(2,3,3)
    dtp = squeeze(sb(:,1,:));
    for isubj = 1:size(dtp,1)
        plot(cTime, dtp(isubj,:), 'LineWidth',1);
        hold on
    end
    title('GLM: bkw (single subject)'), xlabel('lag (ms)'), ylabel('bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'YTick',(-.15:.05:.15), 'YLim',[-0.15 0.15], 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% Cross-Correlation (fwd-bkw), single subject
    subplot(2,3,4) 
    dtp = squeeze(sf2(:,1,:)-sb2(:,1,:));
    for isubj = 1:size(dtp,1)
        plot(cTime, dtp(isubj,:), 'LineWidth',1);
        hold on
    end
    title('Correlation: fwd-bkw (single subject)'), xlabel('lag (ms)'), ylabel('fwd minus bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% Cross-Correlation (fwd), single subject
    subplot(2,3,5)
    dtp = squeeze(sf2(:,1,:));
    for isubj = 1:size(dtp,1)
        plot(cTime, dtp(isubj,:), 'LineWidth',1);
        hold on
    end
    title('Correlation: fwd (single subject)'), xlabel('lag (ms)'), ylabel('fwd sequenceness')
    set(gca, 'XTick',(0:100:600), 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background

    %% Cross-Correlation (bkw), single subject
    subplot(2,3,6) 
    dtp = squeeze(sb2(:,1,:));
    for isubj = 1:size(dtp,1)
        plot(cTime, dtp(isubj,:), 'LineWidth',1);
        hold on
    end
    title('Correlation: bkw (single subject)'), xlabel('lag (ms)'), ylabel('bkw sequenceness')
    set(gca, 'XTick',(0:100:600), 'TickLength', [0.01 0.01], 'TickDir', 'in',...
    'LineWidth',1, 'FontSize', 16, 'Fontname','Arial', 'color', 'none', 'box', 'off'); % 'color', 'none': for transparent background
    
end

