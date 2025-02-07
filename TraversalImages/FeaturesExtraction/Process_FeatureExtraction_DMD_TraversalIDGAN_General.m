function [ feature, ALS, boundary, DryMassFluc ] = Process_FeatureExtraction_DMD_TraversalIDGAN( i,sysConfig, phase, mask, scale )
%Inut parameters
% sysConfig               :  Structure containing experimental parametersStarting parallel pool (parpool) using the 'local' profile ...
% phase                     :  Phase map
% qcellt                      :  Intensity map
% fluo                        :  Fluorescence signal
% B                          :  Boundary information ( unit in pixel )
% mask                      :  Mask of cell
% scale                       :  Pixel to meter ratio
% Output parameters
% feature                    :  Feature extracted from phase/intensity map
% ALS                       :  Angular light scattering profile
% boundary                 :  Boundary of mask ( unit in meter )


%% Default setting
feature = []; ALS = []; boundary = [];

% Define detection range ( Diameter of detection spot )
system_res = 0.61*sysConfig.wavelength/sysConfig.NA;
LocalRange = round(system_res/scale.unit_xs);
LocalRange = LocalRange+1-rem(LocalRange,2);

%% Features from mask
%Pix2Micrometer = sqrt(scale.unit_xs*scale.unit_ys);                      % Convertion constant (pixel->micro-meter)
maskprops = regionprops(mask,'Area','Centroid','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation','Perimeter');
[ xLoc, yLoc ] = meshgrid( 1:size(mask,2), 1:size(mask,1));
CenterOfMask = fliplr(maskprops.Centroid);
%     CellWidth = maskprops.MinorAxisLength;
%     CellLength = maskprops.MajorAxisLength;
CellWidth = sum(max(mask,[],2));
CellLength = sum(max(mask,[],1));
aspectRatio = CellWidth/CellLength;
eccentricity = maskprops.Eccentricity;
if maskprops.Orientation >= 0
    orient = 90-maskprops.Orientation;
else
    orient = -90-maskprops.Orientation;
end
Pix2Micrometer= scale.unit_xs; %approx

% Projected area ( on imaging plane, unit in um^2 )
area_pix = maskprops.Area;
proj_area = area_pix*Pix2Micrometer^2;

if(CellLength > CellWidth)
    CellDimension=CellLength; else CellDimension=CellWidth; end
% Volume ( Elliptical shape (Compressed sphere) assumption, unit in um^3 )
VolApp = 4/3*pi*(CellWidth/2)^2*(CellLength/2)*Pix2Micrometer^3;

%%Computation of Volume of cells by slicing method
C = num2cell(mask,1);
h=1;
VolVector=[];
% for k = 1:numel(C)
%     if(sum(C{1,k})~=0)
%         vec = C{k};
%         radius=sum(C{1,k})/2;
%         VolAcc=pi*radius^2*h*Pix2Micrometer^3;
%         VolVector=[VolVector,VolAcc];
%     end
% end
% VolAcc=sum(VolVector);

% Circularity (unitness)
%Boundary = B{1};
perimeter = maskprops.Perimeter;
circularity = 4*pi*area_pix/(perimeter+pi)^2;
% deformation=1-(2*sqrt(pi*area_pix)/perimeter);
deformation=1-circularity;
%boundary(:,2) = Boundary(:,2)*scale.unit_xs;
%boundary(:,1) = Boundary(:,1)*scale.unit_ys;

% Feature set (mask)
feature_mask = [ proj_area, VolApp, circularity, deformation,eccentricity, aspectRatio, orient ]; %7



    
%image1=imagesc(mask);colormap(hot);daspect([1 1 1]);axis off;
%saveas(image1,strcat('C:\Users\Kevin Tsia\idgan-master\outputs\Traversal\Mask\',num2str(i),'.png'));
%image3=imagesc(phase); colormap(jet); daspect([1 1 1]); axis off;
%% Features from bright-field contrast
[ AngleMap, distanceMap ] = cart2pol(xLoc-CenterOfMask(2), yLoc-CenterOfMask(1));
maskedge = bwboundaries(mask);maskedge = maskedge{1};
edgeDis = sqrt(sum(bsxfun(@minus,maskedge,CenterOfMask).^2,2));
meanEdgeDis = mean(edgeDis);
distanceMap = distanceMap./meanEdgeDis;

%    amp = sqrt(qcellt);
%    cell_body = amp(mask==1);

% Absorption density
%    absop_density = sum(max(cell_body)-cell_body)/proj_area;

% Statistics from Ampitude map
%    Amp_var = var(cell_body(:));
%    Amp_skewness = skewness(cell_body(:),0);
%    Amp_kurtosis = kurtosis(cell_body(:),0);
%    Amp_range = range(cell_body);

% Maximum Ampitude
%    peak_amp = min(cell_body);

% Maximum absorption
%    peak_absorp = max(max(cell_body)-cell_body);

% Feature set (Bright-field)
%    feature_bf = [ absop_density, Amp_var, Amp_skewness, Amp_kurtosis, peak_amp, peak_absorp, Amp_range ]; %8- 14
%    feature = [feature, feature_bf];

%% Features from BF texture
% coarse texture (Entrpopy)
%    bf_entropy = entropyfilt(qcellt, ones(LocalRange*2-1));
%    bfEntr_cell = bf_entropy(mask == 1);

%    bfEntr_mean = mean(bfEntr_cell);
%    bfEntr_var = var(bfEntr_cell, 0);
%    bfEntr_skew = skewness(bfEntr_cell,0);
%    bfEntr_kur = kurtosis(bfEntr_cell,0);
%    bfEntr_range = range(bfEntr_cell);

%    bfEntr_peak = max(bfEntr_cell);
%    bfEntr_min = min(bfEntr_cell);

%    regProp = regionprops(mask, bf_entropy,'WeightedCentroid');
%    bfEntr_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
%    bfEntr_distance = sumabs(bf_entropy.*mask.*distanceMap)/proj_area;

%    feature_bftextcoarse = [bfEntr_mean, bfEntr_var, bfEntr_skew, bfEntr_kur, bfEntr_range, bfEntr_peak, bfEntr_min, bfEntr_CenDis, bfEntr_distance ]; %15-23

% Fine texture
%    bf_std = stdfilt(qcellt, ones(LocalRange));
%    bfstd_cell = bf_std(mask == 1);

%   bfstd_mean = mean(bfstd_cell);
%   bfstd_var = var(bfstd_cell, 0);
%   bfstd_skew = skewness(bfstd_cell,0);
%   bfstd_kur = kurtosis(bfstd_cell,0);
%   bfstd_range = range(bfstd_cell);

%   bfstd_peak = max(bfstd_cell);
%   bfstd_min = min(bfstd_cell);

%   regProp = regionprops(mask, bf_std,'WeightedCentroid');
%   bfstd_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
%   bfstd_distance = sumabs(bf_std.*mask.*distanceMap)/proj_area;
%    feature_bftextfine = [bfstd_mean, bfstd_var, bfstd_skew, bfstd_kur, bfstd_range, bfstd_peak, bfstd_min, bfstd_CenDis bfstd_distance]; %24-32

% Fibermetric
%     bfFib = fibermetric(qcellt,'ObjectPolarity','dark');
%     % bfFib_cell = bfFib(mask==1);
%     regProp = regionprops(mask, bfFib,'WeightedCentroid');
%     bfFib_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
%     bfFib_distance = sumabs(bfFib.*mask.*distanceMap)/proj_area;
%     bfFib_75p = sum(bfFib(mask == 1) > 0.75)/area_pix;
%     bfFib_50p = sum(bfFib(mask == 1) > 0.5)/area_pix;
%     feature_bftextfib = [bfFib_CenDis, bfFib_distance, bfFib_75p, bfFib_50p]; %33-36
%
%     feature = [feature, feature_bftextcoarse, feature_bftextfine, feature_bftextfib];

%% Features for focus determination
%     Grad = imgradient(qcellt);
%     GradPh = imgradient(phase);
%     FocusFactor1 = mean(Grad(mask==1));
%     [maxLoc(2), maxloc(1)] = find(Grad==max(Grad(mask==1)));
%     FocusFactor2 = mean(GradPh(mask == 1));
%     percen = prctile(Grad(:),90);
%     FocusFactor1a = mean(Grad(Grad>percen));
%     FocusFactor2a = mean(GradPh(GradPh>percen));
%     percen = prctile(Grad(:),75);
%     FocusFactor1b = mean(Grad(Grad>percen));
%     FocusFactor2b = mean(GradPh(GradPh>percen));
%     FocusFactor3 = var(qcellt(mask==1))/mean(qcellt(mask==1));
%     bgStab = var(qcellt(mask==0));
%     feature_focus = [FocusFactor1, FocusFactor1a, FocusFactor1b, FocusFactor2, FocusFactor2a, FocusFactor2b, FocusFactor3, bgStab]; %37-44
%     feature = [feature, feature_focus];


%% Features from quantitative phase contrast
%if ~isempty(phase)
    % Dry mass analysis (From refractive index map)
    % VolMaptry = (CellWidth/2)^2-(xLoc-CenterOfMask(2)).^2-(aspectRatio*(yLoc-CenterOfMask(1))).^2;
    VolMap = (CellDimension/2)^2-(yLoc-CenterOfMask(1)).^2-(aspectRatio*(xLoc-CenterOfMask(2))).^2;
    %
    %     CellVolMap = real(2*sqrt(mask.*VolMap))*Pix2Micrometer;  % Unit in meter
    %     RIMap = phase./CellVolMap/2/pi*sysConfig.wavelength;
    %     RIMap((abs(RIMap)==NaN))=0;
    %     DryMassDensityMap = RIMap/0.2;
    %     dry_mass_density = mean(DryMassDensityMap(mask==1));     % Unit in pg/fl = pico-gram per femto-liter
    %     dry_mass = dry_mass_density*VolAcc;
    %     dry_mass_var = var(DryMassDensityMap(mask==1));
    %     dry_mass_skewness = skewness(DryMassDensityMap(mask==1), 0);
    %
    %
    
    CellVolMap = real(2*sqrt(mask.*VolMap))*Pix2Micrometer;  % Unit in meter
    RIMap = phase./CellVolMap/2/pi*sysConfig.wavelength;
    RIMap((abs(RIMap)==inf))=0;
    RIMap(isnan(abs(RIMap)))=0;
    
    DryMassDensityMap = RIMap/0.5;
    %dry_mass_density = mean(DryMassDensityMap(mask==1));     % Unit in pg/fl = pico-gram per femto-liter
    %dry_mass = dry_mass_density*VolAcc;
    %dry_mass_var = var(DryMassDensityMap(mask==1));
    %dry_mass_skewness = skewness(DryMassDensityMap(mask==1), 0);
    %%DMD Fluctuations
    %FitMap=SurfFit(DryMassDensityMap,i);
    %Diff_Fluc=DryMassDensityMap-FitMap;
    %FiltOutline=Diff_Fluc.*mask;
    %DryMassFluc=stdfilt(FiltOutline);
    %thresh=im2bw(DryMassFluc,0.01);
    %thresh2=imcomplement(thresh);
    %thresh3=thresh2.*mask;
    %DryMassFluc=thresh3.*DryMassFluc;
    % DMVFib_map = fibermetric(DryMassFluc)
    %DMDVMean=mean(DryMassFluc(:));
    %DMDVVar=var(DryMassFluc(:));
    %DMDVSkew=skewness(DryMassFluc(:));
    %DMDVKurt=kurtosis(DryMassFluc(:));
    
    %          saveas(image,strcat('C:\Users\eee\Documents\MATLAB\TestCodeforVariance\GroundTruthSurfaceFitting\DMV44',num2str(i),'.png'));
    
    
    % end
    % DMVFib_map = entropyfilt(DryMassFluc,ones(LocalRange*2-1));
    %DMVFib_cell = DMVFib_map(mask==1);
    %      B = bwboundaries(DryMassDensityMap);
    %      B=B{1};
    %      BInt=[];
    %      for i=1:size(B,1)
    %         BInt(i)=DryMassDensityMap(B(i,1),B(i,2));
    %      end
    %   %DMDFluc=mean(DryMassFluc(:));
    
    
    DryMassFluc=0;
    %feature_DM = [ dry_mass, dry_mass_density, dry_mass_var, dry_mass_skewness]
    %feature_Fit=[DMDVMean,DMDVVar, DMDVSkew,DMDVKurt]; %45-52
    %masked_phase = mask.*phase;
    % Statistics from phase map
    phase_body = phase(mask==1);
    phase_var = var(phase_body(:));
    phase_skewness = skewness(phase_body(:),0);
    phase_kurtosis = kurtosis(phase_body(:),0);
    phase_range = range(phase_body);
    peak_phase = max(phase_body);
    min_phase = min(phase_body);
    
    regProp = regionprops(mask, phase,'WeightedCentroid');
    phase_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
    
    feature_qp = [ peak_phase, phase_var, phase_skewness, phase_kurtosis, phase_range, min_phase, phase_CenDis];%53-59
    
    
    
    % Texture analysis on phase map
    
    % Cell contrast
    contrast_map = stdfilt(phase, ones(LocalRange));
    contrast_cell = contrast_map(mask==1);
    

   % image2=imagesc(contrast_map);colormap(hot);daspect([1 1 1]); axis off;
   % saveas(image2,strcat('C:\Users\Kevin Tsia\idgan-master\outputs\Traversal\Fit\',num2str(i),'.png'));
    
    texture = mean(contrast_cell);
    texture_std = std(contrast_cell);
    texture_skew = skewness(contrast_cell,0);
    texture_kurt = kurtosis(contrast_cell,0);
    regProp = regionprops(mask, contrast_map, 'WeightedCentroid');
    texture_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
    texture_distance = sumabs(contrast_map.*mask.*distanceMap)/proj_area;
    feature_DC = [ texture, texture_std, texture_skew, texture_kurt, texture_CenDis, texture_distance];%60-65
    
    % Coarse texture
    qpiEntr_map = entropyfilt(phase,ones(LocalRange*2-1));
    qpiEntr_cell = qpiEntr_map(mask==1);
    
    qpiEntr_mean = mean(qpiEntr_cell);
    qpiEntr_var = var(qpiEntr_cell);
    qpiEntr_skew = skewness(qpiEntr_cell);
    qpiEntr_kurt = kurtosis(qpiEntr_cell);
    regProp = regionprops(mask, qpiEntr_map, 'WeightedCentroid');
    qpiEntr_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
    qpiEntr_distance = sumabs(qpiEntr_map.*mask.*distanceMap)/proj_area;
    feature_qpCoarseText = [qpiEntr_mean, qpiEntr_var, qpiEntr_skew, qpiEntr_kurt, qpiEntr_CenDis, qpiEntr_distance]; %66-71
    
    % Fibrous texture
    qpiFib_map = fibermetric(phase);
    regProp = regionprops(mask, qpiFib_map,'WeightedCentroid');
    qpiFib_CenDis = sqrt(sum((regProp.WeightedCentroid - maskprops.Centroid).^2))*Pix2Micrometer;
    qpiFib_distance = sumabs(qpiFib_map.*mask.*distanceMap)/proj_area;
    qpiFib_75p = sum(qpiFib_map(mask == 1) > 0.75)/area_pix;
    qpiFib_50p = sum(qpiFib_map(mask == 1) > 0.5)/area_pix;
    
    feature_qpFibText = [qpiFib_CenDis, qpiFib_distance, qpiFib_75p, qpiFib_50p];%72-75
    
    % Spatial distribution of phase
    % Distance distribution
     distanceMap = distanceMap*Pix2Micrometer; %  [ B, I ] = sort(distanceMap(:)); figure; plot(B, masked_phase(I))
     SPhaseMean_R = sumabs(phase.*distanceMap)/sumabs(phase);
     SPhaseVar_R = sumabs(phase.*distanceMap.^2)/(sumabs(phase)-1);
     SPhaseSkew_R = sumabs(phase.*distanceMap.^3)/(sumabs(phase)-1)/SPhaseVar_R;
    % Angle distribution
    [ angleList, List ] = sort(AngleMap(:));
    phaseList = phase(List);
    [ angleList, ListOrder ] = unique(angleList);
    phaseList = phaseList(ListOrder);
    AngularFreq = linspace(min(angleList(:)),max(angleList(:)), 3600);
    phaseInAngle = interp1(angleList, phaseList, AngularFreq, 'nerest');
    phaseInAngleFreq = fftshift(abs(fft(phaseInAngle)));
    SPhaseVar_ang = sum(phaseInAngleFreq.*AngularFreq.^2)/(sum(phaseInAngleFreq)-1);
    SPhaseKurt_ang = sum(phaseInAngleFreq.*AngularFreq.^4)/(sum(phaseInAngleFreq)-1)/SPhaseVar_ang;
    feature_qpDistri = [SPhaseMean_R, SPhaseVar_R, SPhaseSkew_R, SPhaseVar_ang, SPhaseKurt_ang];%76-80
    
    % Feature set (Quantiataive phase)
    feature = [feature, feature_mask, feature_qp, feature_DC, feature_qpCoarseText, feature_qpFibText, feature_qpDistri ];
    
    %% Angular light scattering (ALS)
    
    %win_size = 60; % FOV in um
    %pad_size = floor([win_size/scale.unit_ys, win_size/scale.unit_xs]);
    %img_shift = floor((pad_size-size(mask))/2);
    
    
    
    %    pad_amp = zeros(pad_size);
    %    pad_amp((1:size(mask,1))+img_shift(1), (1:size(mask,2))+img_shift(2)) = amp.*(mask==1);
    %     pad_phase = zeros(pad_size);
    %     pad_phase((1:size(mask,1))+img_shift(1), (1:size(mask,2))+img_shift(2)) = phase.*(mask==1);
    %
    %     ImgWidth = size(pad_amp, 1);
    %     E_field = pad_amp.*exp(1i*pad_phase);
    %
    % Fourier transfore via projection slice theorm
    %     Radon_field = radon(E_field, 0:1:179);
    %     FieldLength = size(Radon_field,1);
    %     Radon_field = Radon_field(floor((FieldLength+1)/2)-floor(ImgWidth/2):floor((FieldLength+1)/2)+ceil(ImgWidth/2)-1,:);
    %     hamming1d = hamming(ImgWidth, 'periodic');
    %     BipolarFourierField = fft(bsxfun(@times, Radon_field, hamming1d), [], 1);
    %     UnipolarFourierField = BipolarFourierField(1:floor(size(BipolarFourierField,1)/2),:);
    %     ALSlog = log10(UnipolarFourierField.*conj(UnipolarFourierField));
    %     signal = mean(ALSlog, 2);
    %
    % Corresponding scattering angle
    %     qx = 2*pi/sysConfig.FOV*(0:floor(size(BipolarFourierField,1)/2)-1);
    %     angle = 2*asin(qx*sysConfig.wavelength/4/pi)*180/pi;
    
    % Limitation of scatter angle by finite NA
    %     angleLimit = asin(sysConfig.NA)*180/pi;
    %     ALSangle = angle(angle<=angleLimit);
    %     ALSsignal = signal(angle<=angleLimit);
    
    %     % 1D interpolation
    %     interp_angle = linspace(0, angleLimit, 50);
    %     ALS = interp1(ALSangle, ALSsignal, interp_angle);
    

%% Fluorescent measurement
% if ~isempty(fluo)
% %     time_mask = interp1(1:size(mask,1), single(any(mask,2)), linspace(1, size(mask, 1), length(fluo)), 'nearest');
%     time_mask = any(mask,2);
%     Fluo_masked = fluo(time_mask==1,:);
%     Fluo_Height_ch1 =  max(Fluo_masked(:,1));
%     Fluo_Area_ch1 = sum(Fluo_masked(:,1));
%     Fluo_Density_ch1 = Fluo_Area_ch1/vol;
%     if ~isempty(phase)
%         phaseProj = sum(phase,2);
%         phaseProj = phaseProj(time_mask==1);
%         FluoPhaseCorr = corrcoef(Fluo_masked(:,1), phaseProj);
%         if numel(FluoPhaseCorr)>1
%             FluoPhaseCorr_ch1 = FluoPhaseCorr(2);
%         end
%     else
%         FluoPhaseCorr_ch1 = NaN;
%     end
%     feature_fluo = [Fluo_Height_ch1, Fluo_Area_ch1, Fluo_Density_ch1, FluoPhaseCorr_ch1];
%
%     if size(fluo,2)>=2
%         Fluo_Height_ch2 =  max(Fluo_masked(:,2));
%         Fluo_Area_ch2 = sum(Fluo_masked(:,2));
%         Fluo_Density_ch2 = Fluo_Area_ch2/vol;
%         if ~isempty(phase)
%             FluoPhaseCorr = corrcoef(Fluo_masked(:,2), phaseProj);
%             if numel(FluoPhaseCorr)>1
%                 FluoPhaseCorr_ch2 = FluoPhaseCorr(2);
%             end
%         else
%             FluoPhaseCorr_ch2 = NaN;
%         end
%
%         feature_fluo = [feature_fluo, Fluo_Height_ch2, Fluo_Area_ch2, Fluo_Density_ch2, FluoPhaseCorr_ch2];
%     end
%     % Feature set (Fluorescence)
%     feature = [feature, feature_fluo];

%end

end