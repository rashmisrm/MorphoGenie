folderMain=pwd
Images=strcat(folderMain,'\Images\')%Path to Image Folder
Masks=strcat(folderMain,'\Masks\')%Path to Image Folder


FileList = dir(Images);
Subset = 0;
imcount = 0

for Im = 1:size(FileList, 1)
    if contains(FileList(Im).name, '.tif')
        [filepath,name,ext] = fileparts(FileList(Im).name);
        
        
        Image1=imread(strcat(Images, FileList(Im).name));
        Mask1=imread(strcat(Masks, name,'_label.tif'));
        

        %MaskM=im2double(MaskM);        %MaskM=im2double(MaskM);
        [CentIm1, Centroids1]=GetCentImages(Image1, Mask1);
        SaveFolder=strcat(folderMain,'Cropped','/');
        if ~exist(SaveFolder)
            mkdir(SaveFolder)
        end
        for im=1:size(CentIm1,3)
            imwrite(uint8(CentIm1(:,:,im)), strcat(SaveFolder, num2str(imcount),'.png'));
            imcount=imcount+1;
        end

        
    end
end
 
