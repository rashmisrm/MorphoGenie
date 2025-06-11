
 folderMain=pwd
 folderSave=folderMain

 CH1='BBBC022_v1_images_20586w1\';
 CH2='BBBC022_v1_images_20586w2\';
 CH3='BBBC022_v1_images_20586w3\';
 CH4='BBBC022_v1_images_20586w4\';
 CH5='BBBC022_v1_images_20586w5\';


RefFolder=strcat(folderMain,'\', CH1) %Reference folder is the nucleii image
 
folderSave1=strcat(folderSave)


%chdir(RefFolder);
FileList1 = dir(RefFolder);
FileList2 = dir(strcat(folderMain,'\',CH2));
FileList3 = dir(strcat(folderMain,'\',CH3));
FileList4 = dir(strcat(folderMain,'\',CH4));
FileList5 = dir(strcat(folderMain,'\',CH5));


Subset = 0;
imcount=1;
for Im = 1:size(FileList1, 1)
    if contains(FileList1(Im).name, '.tif')
        [filepath1,name1,ext] = fileparts(FileList1(Im).name)
        [filepath2,name2,ext] = fileparts(FileList2(Im).name)
        [filepath3,name3,ext] = fileparts(FileList3(Im).name)
        [filepath4,name4,ext] = fileparts(FileList4(Im).name)
        [filepath5,name5,ext] = fileparts(FileList5(Im).name)
        
        treatref=name1(1:11)
        
        Image1=imread(strcat(folderMain,'\', CH1,FileList1(Im).name));
        
        Mask1=imread(strcat(folderMain,'\',CH1, '/Masks/',name1,'_label.tif'));
        Objects = unique(Mask1(:));
        NumCells=size(Objects)
        
        if NumCells(1) > 30

            [CentIm1, Centroids1]=GetCentImages(Image1, Mask1, name1);

            Image2=imread(strcat(folderMain,'\',CH2,FileList2(Im).name));
            Mask2=imread(strcat(folderMain,'\',CH2, '/Masks/',name2,'_label.tif'));
            [CentIm2, Centroids2]=GetCentImages(Image2, Mask2, name1);
        
            Image3=imread(strcat(folderMain,'\',CH3,FileList3(Im).name));
            Mask3=imread(strcat(folderMain,'\',CH3, '/Masks/',name3,'_label.tif'));
            [CentIm3, Centroids3]=GetCentImages(Image3, Mask3, name1);
        
            Image4=imread(strcat(folderMain,'\',CH4,FileList4(Im).name));
            Mask4=imread(strcat(folderMain,'\',CH4, '/Masks/',name4,'_label.tif'));
            [CentIm4, Centroids4]=GetCentImages(Image4, Mask4, name1);
        
            Image5=imread(strcat(folderMain,'\',CH5,FileList5(Im).name));
            Mask5=imread(strcat(folderMain,'\',CH5,'/Masks/', name5,'_label.tif'));
            [CentIm5, Centroids5]=GetCentImages(Image5, Mask5, name1);
        
        %%COMPARE CENTROIDS FOR 3 CHANNELS
        for ch1=1:size(Centroids1)
            rowCh1=Centroids1.Centroid(ch1,:)
            for ch2=1:size(Centroids2)
                rowCh2=Centroids2.Centroid(ch2,:)
                if abs(rowCh1(1)- rowCh2(1)) <15 && abs(rowCh1(2)- rowCh2(2)) < 15
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

                                       
                                       if ~exist(strcat(folderSave1,'/Ch1/',treatref,'/Ch/'))
                                           mkdir(strcat(folderSave1,'/Ch1/',treatref,'/Ch/'));
                                       end

                                        
                                       if ~exist(strcat(folderSave1,'/Ch2/',treatref,'/Ch/'))
                                           mkdir(strcat(folderSave1,'/Ch2/',treatref,'/Ch/'));
                                       end

                                       if ~exist(strcat(folderSave1,'/Ch3/',treatref,'/Ch/'))
                                           mkdir(strcat(folderSave1,'/Ch3/',treatref,'/Ch/'));
                                       end

                                       
                                       if ~exist(strcat(folderSave1,'/Ch4/',treatref,'/Ch/'))
                                           mkdir(strcat(folderSave1,'/Ch4/',treatref,'/Ch/'));
                                       end

                                       if ~exist(strcat(folderSave1,'/Ch5/',treatref,'/Ch/'))
                                           mkdir(strcat(folderSave1,'/Ch5/',treatref,'/Ch/'));
                                        end

                                        imwrite(Ch1Im, strcat(folderSave1,'/Ch1/', treatref,'/Ch/', num2str(imcount),'.png'))
                                        imwrite(Ch2Im, strcat(folderSave1,'/Ch2/', treatref,'/Ch/', num2str(imcount),'.png'))
                                        imwrite(Ch3Im, strcat(folderSave1,'/Ch3/', treatref,'/Ch/', num2str(imcount),'.png'))
                                        imwrite(Ch4Im, strcat(folderSave1,'/Ch4/', treatref,'/Ch/', num2str(imcount),'.png'))
                                        imwrite(Ch5Im, strcat(folderSave1,'/Ch5/', treatref,'/Ch/', num2str(imcount),'.png'))

                                        imcount=imcount+1;
                                    end
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

        if ~exist(int2str(name1))
           %mkdir(strcat(folderMain,name));
        end
        
    end
end
 
