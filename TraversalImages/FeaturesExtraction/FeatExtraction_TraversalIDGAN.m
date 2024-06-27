function [T]= Looping_DVD_ADQ7_process_interface(folder_name, dim)

%function FeatExtraction(folder)
%function [VarMatRGB] = VarianceMat(ImageRGB,L,W)
%% System parameters  %%

%sysConfig.channel = {[4 2 1 3]};                  % Channel match -- [ Right, Left, Top, Bottom ]
%sysConfig.pumpRate = 1;                      % Volumatric flow rate (ml/hr)
%sysConfig.channel_size = [20 22];                 % Microfluidic channel dimension ([width height] in um)
%sysConfig.FOV = 30;                               % Width of FOV ( in micrometer )
%sysConfig.PSRgain = 1;                            % Pixel-super-resolution (Resolution shift from y-axis to x-axis

%% Constants and pre-set variables %%
%sysConfig.cellsize = [2, 70];                     % Manual define cell size window, below or above which will be discarded (Radius, in micron)
%sysConfig.mediumRI = 1.33;                        % Refractive index of flow medium (e.g. PBS)
%sysConfig.systshift = [0, 0];                     % Systematic croping shift (Y,X)
%sysConfig.phasecorr = 1;                          % Correction constant for quantitative phase
%sysConfig.rep_time = 1000/12.08;                  % Pulse repetition rate ( Before 01/2016: 86.1, After: 84.71)
%sysConfig.samplingMultiple = 336;                 % 336
%sysConfig.disper = 1.78;                          % Dispersion parameter (ns/nm)
%sysConfig.conv_fac = 181;                        % Wavelength to Space mapping parameter (NA dependent)
sysConfig.wavelength = 0.532;                     % Center wavelength of imaging source (um)
sysConfig.NA = 0.6;                              % Numerical aperature
%sysConfig.Delay = [ 17.369, 34.035, 52.944 ];     % Length of dealy lines (in ns)
sysConfig.manualscale = 50;


% %% Calibrate x,y-axis %%
%     noflines = 181;
%     cropFac = 0.8;
%     scale.length_y = 2*cropFac*sysConfig.FOV;
%     linFlowRate = sysConfig.pumpRate/3.6e9/(prod(sysConfig.channel_size)*10^-12);   % Convert volumatric flow rate to linear flow rate
%     scale_range_x = noflines;
%     scale.unit_xs = (sysConfig.rep_time*sysConfig.PSRgain*1e-9*1e6)*linFlowRate;     % where 1e6 convert the space to um
%     scale.unit_ys = (sysConfig.rep_time*sysConfig.PSRgain*1e-9*1e6)*linFlowRate;
%     scale.length_x = scale_range_x*scale.unit_xs;
featureCells=[];DMDFlucNormArray=[];
VolApp=[]; k=1;
%%
cellline='GroundTruth'; %Choose normal for Non-Deformed cells or (mcf or thp or drug)
cellline='IDGANTrav'
celltype='CellCycleFaced'; %Choose NewDataset or OldDataset
if(strcmp(cellline,'thp'))
    folder='C:\Results_ATOM_OC_RTDC_16012019_processed\';
    scale.unit_ys=0.28; scale.unit_xs=0.28;
    AreaThresh=550;
elseif(strcmp(cellline,'thpnew'))
    folder='C:\Results_ATOM_OC_20190712_THP1_straight_processed\';
    % folder='C:\Results_ATOM_OC_20190712_THP1_deform_processed\'
    scale.unit_ys=0.2128; scale.unit_xs=0.2128;
    AreaThresh=550;
elseif(strcmp(cellline,'mcf'))
    folder='C:\Results_ATOM_OC_RTDC_11Apr_MCF7_processed\cells\';
    %folder='C:\Results_ATOM_OC_RTDC_11Apr_MCF7_processed\cells\';
    scale.unit_ys=0.35; scale.unit_xs=0.35;
    AreaThresh=700;
elseif(strcmp(cellline,'normal'))
    folder='C:\THP-1\';
    scale.unit_ys=0.1250; scale.unit_xs=0.1250;
    AreaThresh=500;
elseif(strcmp(cellline,'drug'))
    %folder='C:\Drug Experiments\DeformedControl\';
    % folder='C:\Results_ATOM_OC_20190711_MCF7Vin_deform_processed\';
    folder='C:\Drug Experiments\DeformedDrug\';
    % folder='C:\Drug Experiments\NonDeformedControl\'
    % folder='C:\Drug Experiments\DeformedDrug\';
    scale.unit_ys=0.319; scale.unit_xs=0.319;
    AreaThresh=600;
elseif(strcmp(cellline,'MB231'))
    folder="C:\Results_ATOM_OC_20190719_MB231deform1_processed\";
    
    %     folde="C:\Results_ATOM_OC_20190719_MB231deform2_processed\";
    %     folder="C:\Results_ATOM_OC_20190719_MB231straight_processed\";
    %     folder="C:\Results_ATOM_OC_20190724_MB231deform_processed\";
    %     folder="C:\Results_ATOM_OC_20190724_MB231straight_processed\";
    %     folder="C:\Results_ATOM_OC_20190724_MB231Vindeform_processed\"
    %     folder="C:\Results_ATOM_OC_20190724_MB231Vinstraight_processed\";
    
    %folder='C:\Results_ATOM_OC_20190712_MB231_straight_processed\';
    %folder='C:\Results_ATOM_OC_RTDC_11Apr_MCF7_processed\cells\';
    scale.unit_ys=0.2128; scale.unit_xs=0.2128;
    AreaThresh=500;
    
elseif(strcmp(cellline,'GroundTruth'))
    %folder='C:\Drug Experiments\DeformedControl\';
    % folder='C:\Results_ATOM_OC_20190711_MCF7Vin_deform_processed\';
    folder='C:\GroundTruth\1\images\';
    % folder='C:\Drug Experiments\NonDeformedControl\'
    % folder='C:\Drug Experiments\DeformedDrug\';
    scale.unit_ys=0.1973; scale.unit_xs=0.1973;
    AreaThresh=500;
elseif(strcmp(cellline,'MB231-CellCycle'))
    %folder='E:\Rashmi\mb231-20200907T082526Z-001\mb231\36\G1\'
    load('info.mat')
    load('lookup_table.mat')
elseif(strcmp(cellline,'IDGANTrav'))
    folder=folder_name
    scale.unit_ys=0.15; scale.unit_xs=0.15;

    
end
addpath(folder)
list = dir(fullfile(folder,['*.png']));
%CellFiles=list.name;
%%
    
for i=1:size(list,1)
    cell_file_name=strcat(folder,list(i).name);
    Phase=imread(cell_file_name) ;
   % Phase=imread('untitled.jpg')
    i

            Phase=im2double(Phase);
            cmap = jet(256);
            Phase2d = rgb2ind(Phase, cmap);
            Phase2d=im2double(Phase2d);
            %Intensity=rgb2gray(Phase);
            %Intensity=im2double(Intensity);
            %cell_mask=im2bw(Phase2d,0.3);

            cell_mask=im2bw(Phase,0.3);
            se = strel('disk',4,4);
            cell_mask = imerode(cell_mask,se);

            cell_mask = imerode(cell_mask,se);
            cell_mask = imdilate(cell_mask,se);
            cell_mask = imdilate(cell_mask,se);


            cell_mask=imfill(cell_mask,'holes');
%              se = strel('disk',8,8);
%              cell_mask = imdilate(cell_mask,se);
%              cell_mask = imdilate(cell_mask,se);
%              cell_mask = imdilate(cell_mask,se);
%              cell_mask = imdilate(cell_mask,se);
%              cell_mask = imdilate(cell_mask,se);
% 
%              cell_mask = imerode(cell_mask,se);
%              cell_mask = imerode(cell_mask,se);
%              cell_mask = imerode(cell_mask,se);
%              cell_mask = imerode(cell_mask,se);
%               cell_mask = imerode(cell_mask,se);

           % cell_mask=imcomplement(cell_mask)
            

           %[B,L] = bwboundaries(cell_mask,'noholes');
%             
              maskprops = regionprops(cell_mask,'Area','Centroid','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation','Perimeter');
              if (length(str2mat(maskprops.Centroid)) > 1)
                    se = strel('disk',8,8);
                    cell_mask = imerode(cell_mask,se);
                    cell_mask = imerode(cell_mask,se);
                    cell_mask = imerode(cell_mask,se);
                    cell_mask = imerode(cell_mask,se);
                    
                    cell_mask = imdilate(cell_mask,se);
                    cell_mask = imdilate(cell_mask,se);
                    cell_mask = imdilate(cell_mask,se);
                    cell_mask = imdilate(cell_mask,se);
                    cell_mask=imfill(cell_mask,'holes');
              end
%                
             %j
            [feature, ALS, boundary, DMDFlucNorm ] = Process_FeatureExtraction_DMD_TraversalIDGAN(k, sysConfig, Phase2d, cell_mask, scale );
            featureCells=vertcat(featureCells,feature);
            k=k+1;

        end
        
   % end


   % imagesc(cell_phase);colormap(gca,jet_hd);colorbar;daspect([1 1 1]);




 
% 
   %  end

        %end
    featureList = {}; UnitList = {''};
    Label_mask = {'Area', 'Volume', 'Circularity','Deformation', 'Eccentricity', 'AspectRatio', 'Orientation'};%1-7
    Unit_mask = {'\mum^2', 'fl','', '', '', '','degrees'};
   % Label_bf = {'AttenuationDensity', 'AmplitudeVar', 'AmplitudeSkewness', 'AmplitudeKurtosis', 'PeakAmplitude', 'PeakAbsorption', 'AmplitudeRange'};%8-14
   % Unit_bf = {'1/um^2', '', '', '', '','',''};
    Label_bftextc = {'BFEntropyMean', 'BFEntropyVar', 'BFEntropySkewness', 'BFEntropyKurtosis', 'BFEntropyRange', 'BFEntropyPeak', 'BFEntropyMin', 'BFEntropyCentroidDisplacement', 'BFEntropyRadialDistribution'};%15-23
   % Unit_bftextc = {'','','','','','','','um',''};
  %  Label_bftextf = {'BFSTDMean', 'BFSTDVar', 'BFSTDSkewness', 'BFSTDKurtosis', 'BFSTDRange', 'BFSTDPeak', 'BFSTDMin', 'BFSTDCentroidDisplacement', 'BFSTDRadialDistribution'};%24-32
  %  Unit_bftextf = {'','','','','','','','um',''};
  %  Label_bftextfib = {'BFFiberTextureCentroidDisplacement', 'BFFiberTextureRadialDistribution', 'BFFiberTexturePixelUpperPercentile', 'BFFiberTexturePixelMedian'}; %33-36
  %  Unit_bftextfib = {'um','', '', ''};
    Label_focus = {'Focusfactor1', 'Focusfactor1a', 'Focusfactor1b', 'Focusfactor2','Focusfactor2a','Focusfactor2b', 'Focusfactor3','BackgroundStability'}; %37-44
    Unit_focus = {' ', ' ', ' ', ' ', ' ', ' ',' ',' '};
    
    featureList = [featureList, Label_mask];
    UnitList = [UnitList, Unit_mask ];%count=43
    
    
    Label_DM = {'DryMass', 'DryMassDensity', 'DryMassVar', 'DryMassSkewness'}; %45-48
    Label_DMFit={'FitTextMean','FitTextVar','FitTextSkew','FitTextKurt'}
    
    Unit_DM = {'pg', 'pg/fl', 'pg^2', ''};
    Label_qp = {'PeakPhase', 'PhaseVar', 'PhaseSkewness', 'PhaseKurtosis', 'PhaseRange', 'PhaseMin', 'PhaseCentroidDisplacement'};%49-55
    Unit_qp =  {'rad', 'rad^2', '', '', 'rad', 'rad','um'};
    Label_DMDVar={'DMDVarMean','DMDVarVariance','DMDVarSkewness','DMDVarKurtosis'};
    Unit_DMFit={'','','',''};
    Label_DC = {'DMDContrast1', 'DMDContrast2', 'DMDContrast3', 'DMDContrast4', 'DMDCentroidDisplacement', 'DMDRadialDistribution'};%56-61
    Unit_DC = {'rad', 'rad', '', '','um',''};
    Label_qptextc = {'QPEntropyMean', 'QPEntropyVar', 'QPEntropySkewness', 'QPEntropyKurtosis', 'QPEntropyCentroidDisplacement', 'QPEntropyRadialDistribution'};%62-67
    Unit_qptextc = {'','','','','',''};
    Label_qptextfib = {'QPFiberCentroidDisplacement', 'QPFiberRadialDistribution', 'QPFiberPixelUpperPercentile', 'QPFiberPixelMedian'};%68-71
    Unit_qptextfib = {'um','','',''};
    Label_qpDistri={'MeanPhaseArrangement', 'PhaseArrangementVar', 'PhaseArrangementSkewness', 'PhaseOrientationVar', 'PhaseOrientationKurtosis'};%72-76
    Unit_qpDistri = {'um', 'um^2', '', '', ''};
%     featureList = [featureList, Label_qp, Label_DM,Label_DMDVar, Label_DC, Label_qptextc, Label_qptextfib, Label_qpDistri];
%     UnitList = [UnitList,Unit_qp, Unit_DM, Unit_DMDVar, Unit_DC, Unit_qptextc, Unit_qptextfib, Unit_qpDistri];
    %featureList = [featureList, Label_qp, Label_DM, Label_DMFit Label_DC, Label_qptextc, Label_qptextfib, Label_qpDistri];
    %UnitList = [UnitList,Unit_qp, Unit_DM,Unit_DMFit Unit_DC, Unit_qptextc, Unit_qptextfib, Unit_qpDistri];
    
    featureList = [featureList, Label_qp, Label_DM ,Label_DMDVar,Label_DC, Label_qptextc, Label_qptextfib, Label_qpDistri];
    UnitList = [UnitList,Unit_qp, Unit_DM, Unit_DC, Unit_qptextc, Unit_qptextfib, Unit_qpDistri];
    
    %featureList=cell2table(featureList');
    %featureCells=array2table(featureCells');
    %T = {,featureCells};  
    
FeatureTable=array2table(featureCells);
FeatureTable.Properties.VariableNames=featureList;
%save(cell_file_name,'FeatureTable','-append')
text_file_name=strcat(folder,'G1.csv');
writetable(FeatureTable, text_file_name);
FeatureArray=table2array(FeatureTable)
VarianceAllCol=[]
dim=dim
j=0
featN=43
NormArray=normalize(FeatureArray)

for i=1:featN
    VarianceSingCol=[]

    TempArray=(NormArray(:,i))
    s=1
    e=10;
    for j=1:dim
        Variance=var(TempArray(s:e))
        VarianceSingCol=[VarianceSingCol;Variance]
        s=s+10
        e=e+10;
    end
    VarianceAllCol=[VarianceAllCol VarianceSingCol ]
end
        

VarianceAllCol=normalize(VarianceAllCol)   
rowNorm=[];
for row=1:size(VarianceAllCol,1)
    rowNorm=[rowNorm; normalize(VarianceAllCol(row,:))];
end
T=array2table(VarianceAllCol);    
T.Properties.VariableNames=featureList;
h=heatmap(VarianceAllCol,'Colormap', parula )
h.XData=featureList
ax=gca
ax.Colormap('jet')
ax.XData=featureList
text_file_name=strcat(folder,'VarHeatMap.csv');
writetable(T, text_file_name);
end