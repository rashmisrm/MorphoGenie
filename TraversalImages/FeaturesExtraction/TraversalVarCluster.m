%folder='C:\Users\Rashmi\WorkingLibs\idgan-master_Jan\outputs\TraversalLoop\Benchmarking\Factor\LC-Gray\New\Gamma=6.4\'
%folder='C:\Users\Rashmi\WorkingLibs\idgan-master_Jan\outputs\TraversalLoop\Benchmarking\Factor\LiveCell\Gamma=100\'
%folder='C:\Users\Rashmi\WorkingLibs\idgan-master_Jan\outputs\TraversalLoop\Benchmarking\Beta\LiveCell\Beta=50\'
%folder='C:\Users\Rashmi\WorkingLibs\idgan-master_Jan\outputs\TraversalLoop\Benchmarking\Factor\EMT\Gamma=10\'
%folder='C:\Users\Rashmi\WorkingLibs\idgan-master_Jan\outputs\TraversalLoop\FreshBenchmarking\Factor\EMT\Gamma=10\'
folder='C:\Users\Rashmi\WorkingLibs\idgan-master_Jan\outputs\TraversalLoop\'
Min=1
Max=49
c_dim=10
%thresh_level=0.1   %For EMT and LiveCell and CPA Cellcycle
%thresh_level=0.25   %For LC

%thresh_level=0.1 %(EMT)
thresh_level=0.08 %(LC)
manual_features=35
for n=Min:Max
    folder_no=strcat(folder,num2str(n),'\')
    if(~isempty(folder_no))
        T=FeatExtraction_TraversalIDGAN_General(folder_no, c_dim, manual_features, thresh_level);
    else 
        continue
    end
    %T=FeatExtraction_TraversalIDGAN(strcat(folder,num2str(n),'\'), c_dim)
end
TSum=zeros(c_dim, manual_features)
TAll={}
for n=Min:Max
    T=readtable(strcat(folder,num2str(n),'\','VarHeatMap.csv'));
    T_Array=table2array(T)
    %Check NAN vector
    TNaN = isnan(T_Array);
    VecSum=sum(TNaN);
    for i=1:size(VecSum,2)     
        if(VecSum(i)>0)
            vec=zeros(c_dim,1);
            T_Array(:,i)=vec;
            
        end
    end
    TSum=TSum+abs(T_Array);
    n
end
TSum=TSum/(Max-Min+1)

features=T.Properties.VariableNames

TableSum=array2table(TSum)

TableSum.Properties.VariableNames=features

%TableSum=TableSum(:,any(TableSum))

writetable(TableSum, strcat(folder,'\','DummyNew008.csv'));
% 
% MVal=[]
% % % Automatically remove redundant dimensions
%  MVal=mean(TSum,2)
% % min=min(MVal)
% count=0
% 
% %Select least important dimensions by considering the lowest values of
% %weighted sum of every dimension
% 
% n=3
% 
% while (1)
%     for i =1:size(MVal,1)
%         if MVal(i)==min(MVal)
%             TSum(i,:)=[];
%             MVal=mean(TSum,2)
%             count=count+1
%             if count==n
%                 break
%             end
%             break
%         end
%             if count==n
%                 break
%             end
%     end
%            if count==n
%                 break
%            end
% end
% 
%         
% TSum(1,:)=[]
% TSum(4-1,:)=[]
% 
% TSum(5-2,:)=[]
% %TSum(8-3,:)=[]
% TSum=VarianceAllCol
% Rows=[2, 3, 6, 7, 8, 9, 10]
% [cm] = cb_colormap_gen('Greek')
% 
% cgo = clustergram((TSum),'Standardize','Row','Colormap',cm)
% set(cgo,'ColumnLabels',features,'ColumnLabelsRotate', 45)
% set(cgo,'RowLabels',Rows,'RowLabelsRotate', 0)
% 
% 
% Table=array2table(TSum)
% Table.Properties.VariableNames=features
% 
% 
% %TSum(8-3,:)=[]