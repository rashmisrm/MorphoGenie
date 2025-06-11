function[CentImages, Centroids]=GetCentImages(Image, MaskM, name)
        Objects = unique(MaskM(:));
        
        count=1
        CentroidAll=[]
        ImageAll=[]
        for i =2:size(Objects)
                i
                %figure(1); imagesc(MaskM == Objects(i));
                SegMask= MaskM == Objects(i);
                SegMask=imclearborder(SegMask);
                CC =bwconncomp(SegMask);
                if CC.NumObjects ~=1
                    continue;
                end
                Centroid = regionprops(CC,'centroid');
                Areas = regionprops(CC,'Area');


                %if(Areas(i).Area > 40 && Areas(i).Area < 45840)
                    SegIm=SegMask.*(im2double(Image));
                    row=size(MaskM,2);
                    col=size(MaskM,1);
                    Centerx=abs(row/2);
                    Centery=abs(col/2);
                    Crop=[Centerx-100,Centery-100, 200, 200];

                    filename=sprintf('SegIm%04d',i)
                    ShiftX=Centerx - round(Centroid.Centroid(1));
                    ShiftY=Centery - round((Centroid.Centroid(2)));
                    ShiftedCent = circshift(SegIm,[ShiftY,ShiftX]);
                    B1 = imcrop(ShiftedCent,Crop);


                    %rgbImage = cat(3, B1, B1, B1);
                    Submin = min(B1(:));
                    Submax = max(B1(:));
                    AdjBGSub = uint8( (B1 - Submin)/(Submax-Submin) * 255);
                    Ia = AdjBGSub;
                    figure(1); imagesc(AdjBGSub); colormap gray; axis off; daspect([1 1 1])
                    %saveas(figure, strcat(folderMain,name,'\',int2str(i),'.png'));
                    ImageAll(:,:,count)=(AdjBGSub);
                    count=count+1
                    CentroidAll=[CentroidAll; Centroid];
                    if(isempty(CentroidAll))
                        print('Wait Here')
            %end
                    end
            CentImages=ImageAll;
        end
            Centroids=struct2table(CentroidAll);
        end

