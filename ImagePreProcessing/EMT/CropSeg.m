
 folderMain='Z:/COVID-FTP/EMT/vimentin_img1/1/' %Path to Image Folder
 RefFolder='Z:/COVID-FTP/EMT/vimentin_img1/1/Masks'  %Path to Mask Folder


chdir(folderMain);
FileList = dir(folderMain);
Subset = 0;
imcount = 0

for Im = 1:size(FileList, 1)
    if contains(FileList(Im).name, '.tif')
        [filepath,name,ext] = fileparts(FileList(Im).name)
        
        
        Image1=imread(strcat(folderMain, FileList(Im).name));
        Mask1=imread(strcat(folderMain,'/Masks/', name,'_label.tif'));
        

        %MaskM=im2double(MaskM);        %MaskM=im2double(MaskM);
        [CentIm1, Centroids1]=GetCentImages(Image1, Mask1, name)
        SaveFolder=strcat(folderMain,'Cropped','/')
        if ~exist(SaveFolder)
            mkdir(SaveFolder)
        end
        for im=1:size(CentIm1,3)
            imwrite(uint8(CentIm1(:,:,im)), strcat(SaveFolder, num2str(imcount),'.png'))
            imcount=imcount+1;
        end

        
    end
end
 
