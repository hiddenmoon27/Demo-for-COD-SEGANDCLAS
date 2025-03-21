clear all;
close all; 
clc;

% ---- 1. Camouflage Map Path Setting ----
CamMapPath = 'C:/python/UI/SINetV2/res/OSFormer_Mask';   % Put model results in this folder.
Models = {'SINet-V2'};
modelNum = length(Models);

% ---- 2. Ground-truth Datasets Setting ----
DataPath = 'C:/python/UI/SINetV2/COD10K-v3/Test';
Datasets = {'COD10K','CHAMELEON','CAMO'};
% Datasets = {'COD10K-Amphibian', 'COD10K-Aquatic', 'COD10K-Flying', 'COD10K-Terrestrial'};
% ---- 3. Results Save Path Setting ----
ResDir = './OSEvaluationResults/';
ResName='_result.txt';  % You can change the result name.

Thresholds = 1:-1/255:0;
datasetNum = length(Datasets)-2;

for d = 1:datasetNum
    tic;
    dataset = Datasets{d}   % print current dataset name
    fprintf('- Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
  
    ResPath = [ResDir dataset '-mat/']; % The result will be saved in *.mat file so that you can used it for the next time.
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset ResName];  % The evaluation result will be saved in `../Resluts/Result-XXXX` folder.
    fileID = fopen(resTxt,'w');
    
    for m = 1:modelNum
        model = Models{m}   % print cur model name

        gtPath = [DataPath  '/GT_Object/'];
        camPath = [CamMapPath '/'];
        
        imgFiles = dir([gtPath '*.png']);  
        imgNUM = 2026;
        
        [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));
        
        [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
        
        [Smeasure, wFmeasure, adpFmeasure, adpEmeasure, MAE] = deal(zeros(1,imgNUM));
        
        % Start parallel pool to bring out your computer potentials (for -> parfor)
        for i = 1:imgNUM
            fprintf('> Evaluating(%s Dataset,%s Model): %d/%d\n',dataset, model, i,imgNUM);
            name =  imgFiles(i).name;
            name1=strrep(name,  '.png','.jpg');
            %load gt
            gt = imread([gtPath name]);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            % load camouflaged prediction
            cam = imread([camPath name1]);
            fprintf('%s is in process\n', [camPath name1]);
            % check size
            if size(cam, 1) ~= size(gt, 1) || size(cam, 2) ~= size(gt, 2)
                cam = imresize(cam,size(gt));
                imwrite(cam,[camPath name]);
                fprintf('[Size Mismatching] Error occurs in the path: %s!!!\n', [camPath name]);
            end
            
            cam = im2double(cam(:,:,1));
            
            % normalize CamMap to [0, 1]
            cam = reshape(mapminmax(cam(:)', 0, 1), size(cam));
            % S-meaure metric published in ICCV'17 (Structure measure: A New Way to Evaluate the Foreground Map.)
            Smeasure(i) = StructureMeasure(cam, logical(gt)); 
            % Weighted F-measure metric published in CVPR'14 (How to evaluate the foreground maps?)
            if i>=0
                fprintf('gt 的类型是：%s\n', class(logical(gt)));
                fprintf('cam 的类型是：%s\n', class(cam));
            end
            wFmeasure(i) = original_WFb(cam, logical(gt));
            
            % Using the 2 times of average of cam map as the threshold.
            %threshold =  2 * mean(cam(:));    %To insure max value is less than 1.
            threshold = min(2 * mean2(cam), 1);
            [~, ~, adpFmeasure(i)] = Fmeasure_calu(cam,double(gt),size(gt),threshold);
            
            Bi_cam = zeros(size(cam));
            Bi_cam(cam>threshold)=1;
            adpEmeasure(i) = Enhancedmeasure(Bi_cam, gt);
            
            [threshold_F, threshold_E]  = deal(zeros(1, length(Thresholds)));
            [threshold_Pr, threshold_Rec]  = deal(zeros(1, length(Thresholds)));
            for t = 1:length(Thresholds)
               threshold = Thresholds(t);
               [threshold_Pr(t), threshold_Rec(t), threshold_F(t)] = Fmeasure_calu(cam,double(gt),size(gt),threshold);    
               Bi_cam = zeros(size(cam));
               Bi_cam(cam>threshold)=1;
               threshold_E(t) = Enhancedmeasure(Bi_cam,gt);
            end
            
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Precion(i,:) = threshold_Pr;
            threshold_Recall(i,:) = threshold_Rec;
            
            MAE(i) = mean2(abs(double(logical(gt)) - cam));
             % Print evaluation metrics for the current image
            fprintf('(Dataset:%s; Model:%s) Smeasure:%.3f; wFmeasure:%.3f; MAE:%.3f; adpEm:%.3f; adpFm:%.3f.\n', dataset, model, Smeasure(i), wFmeasure(i), MAE(i), adpEmeasure(i), adpFmeasure(i));
            
            
        end

        column_F = mean(threshold_Fmeasure,1);
        meanFm = mean(column_F);
        maxFm = max(column_F);
        
        column_Pr = mean(threshold_Precion,1);
        column_Rec = mean(threshold_Recall,1);
        
        column_E = mean(threshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        Sm = mean2(Smeasure);
        wFm = mean2(wFmeasure);
        
        adpFm = mean2(adpFmeasure);
        adpEm = mean2(adpEmeasure);
        mae = mean2(MAE);
        
        save([ResPath model],'Sm','wFm', 'mae', 'column_Pr', 'column_Rec', 'column_F', 'adpFm', 'meanFm', 'maxFm', 'column_E', 'adpEm', 'meanEm', 'maxEm');
        fprintf(fileID, '(Dataset:%s; Model:%s) Smeasure:%.3f; wFmeasure:%.3f;MAE:%.3f; adpEm:%.3f; meanEm:%.3f; maxEm:%.3f; adpFm:%.3f; meanFm:%.3f; maxFm:%.3f.\n',dataset,model,Sm, wFm, mae, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm); 
        fprintf('(Dataset:%s; Model:%s) Smeasure:%.3f; wFmeasure:%.3f;MAE:%.3f; adpEm:%.3f; meanEm:%.3f; maxEm:%.3f; adpFm:%.3f; meanFm:%.3f; maxFm:%.3f.\n',dataset,model,Sm, wFm, mae, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm);
    end
    toc;
    
end


