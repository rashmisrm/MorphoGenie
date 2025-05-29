%CellType='A172'
%folderMain='Z:\COVID-FTP\LiveCell_Seg\A172\'
%folderMask='Z:\COVID-FTP\LiveCell_Seg\A172\Masks\'
% 
 folderMain='C:/Users/Rashmi/ProcessedData/PA/'
 RefFolder='C:/Users/Rashmi/ProcessedData/PA/DKC1/'
 %folderMask='Z:\COVID-FTP\CP-VAE\3\Masks\'
% 
% folderMain='Z:\COVID-FTP\CP-VAE\3\'
% folderMask='Z:\COVID-FTP\CP-VAE\3\Masks\'

% folderMain='Z:\COVID-FTP\CP-VAE\2\'
% folderMask='Z:\COVID-FTP\CP-VAE\2\Masks\'
% 
% folderMain='Z:\COVID-FTP\CP-VAE\2\'
% folderMask='Z:\COVID-FTP\CP-VAE\2\Masks\'
%folderMask='Z:\COVID-FTP\CP-VAE\1\Masks\'

chdir(RefFolder);
FileList = dir(RefFolder);
Subset = 0;
imcount=1;
for Im = 1:size(FileList, 1)
    if contains(FileList(Im).name, '.tif')
        [filepath,name,ext] = fileparts(FileList(Im).name)
        
        Image1=imread(strcat(folderMain,'1/',FileList(Im).name));
        Mask1=imread(strcat(folderMain,'1/Masks/', name,'.tif'));
        [CentIm1, Centroids1]=GetCentImages(Image1, Mask1, name);

        Image2=imread(strcat(folderMain,'2/',FileList(Im).name));
        Mask2=imread(strcat(folderMain,'2/Masks/', name,'.tif'));
        [CentIm2, Centroids2]=GetCentImages(Image2, Mask2, name);
        
        Image3=imread(strcat(folderMain,'3/',FileList(Im).name));
        Mask3=imread(strcat(folderMain,'3/Masks/', name,'.tif'));
        [CentIm3, Centroids3]=GetCentImages(Image3, Mask3, name);
        
        Image4=imread(strcat(folderMain,'4/',FileList(Im).name));
        Mask4=imread(strcat(folderMain,'4/Masks/', name,'.tif'));
        [CentIm4, Centroids4]=GetCentImages(Image4, Mask4, name);
        
        Image5=imread(strcat(folderMain,'5/',FileList(Im).name));
        Mask5=imread(strcat(folderMain,'5/Masks/', name,'.tif'));
        [CentIm5, Centroids5]=GetCentImages(Image5, Mask5, name);
        
        %%COMPARE CENTROIDS FOR 3 CHANNELS
        for ch1=1:size(Centroids1)
            rowCh1=Centroids1.Centroid(ch1,:)
            for ch2=1:size(Centroids2)
                rowCh2=Centroids2.Centroid(ch2,:)
                if abs(rowCh1(1)- rowCh2(1)) <15 && abs(rowCh1(2)- rowCh2(2)) < 15
                %if Centroids1.Centroid(ch1,:)
                    for ch3=1:size(Centroids3)
                        rowCh3=Centroids3.Centroid(ch3,:)
                        if abs(rowCh1(1)- rowCh3(1)) <15 && abs(rowCh1(2)- rowCh3(2)) < 15
                            
                        for ch4=1:size(Centroids4)
                            rowCh4=Centroids4.Centroid(ch4,:)
                            if abs(rowCh1(1)- rowCh4(1)) <15 && abs(rowCh1(2)- rowCh4(2)) < 15
                                

                                for ch5=1:size(Centroids5)
                                    rowCh5=Centroids5.Centroid(ch5,:)
                                    if abs(rowCh1(1)- rowCh5(1)) <15 && abs(rowCh1(2)- rowCh5(2)) < 15
                                        Ch1Im=uint8(CentIm1(:,:,ch1));
                                        Ch2Im=uint8(CentIm2(:,:,ch2));
                                        Ch3Im=uint8(CentIm3(:,:,ch3));
                                        Ch4Im=uint8(CentIm4(:,:,ch4));
                                        Ch5Im=uint8(CentIm5(:,:,ch5));

                                       if ~exist(strcat(folderMain,'/Ch1/'))
                                           mkdir(strcat(folderMain,'/Ch1/'));
                                       end
                                        if ~exist(strcat(folderMain,'/Ch2/'))
                                           mkdir(strcat(folderMain,'/Ch2/'));
                                        end
                                       if ~exist(strcat(folderMain,'/Ch3/'))
                                           mkdir(strcat(folderMain,'/Ch3/'));
                                       end
                                       if ~exist(strcat(folderMain,'/Ch4/'))
                                           mkdir(strcat(folderMain,'/Ch4/'));
                                       end
                                       if ~exist(strcat(folderMain,'/Ch5/'))
                                           mkdir(strcat(folderMain,'/Ch5/'));
                                       end                                       
                                        imwrite(Ch1Im, strcat(folderMain,'/Ch1/',num2str(imcount),'.png'))
                                        imwrite(Ch2Im, strcat(folderMain,'/Ch2/',num2str(imcount),'.png'))
                                        imwrite(Ch3Im, strcat(folderMain,'/Ch3/',num2str(imcount),'.png'))
                                        imwrite(Ch4Im, strcat(folderMain,'/Ch4/',num2str(imcount),'.png'))
                                        imwrite(Ch5Im, strcat(folderMain,'/Ch5/',num2str(imcount),'.png'))

                                        imcount=imcount+1
                                    end
                                end
                            end
                        end
                    end
                end
            end
            end
            
        end
        %MaskM=im2double(MaskM);

        if ~exist(int2str(name))
           %mkdir(strcat(folderMain,name));
        end
        
    end
end
 
