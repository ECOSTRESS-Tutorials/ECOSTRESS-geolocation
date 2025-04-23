import os
import re
import gc
import time
import cv2
import csv
import numpy as np
import concurrent.futures

from osgeo import gdal
from datetime import datetime
from pandas import DataFrame
from multiprocessing import freeze_support
from scipy.ndimage import binary_hit_or_miss
from skimage import feature, morphology, measure

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

## Andrew Alamillo, Jet Propulsion Laboratory, California Institute of Technology, Pasadena, California, USA
## Acknowledgements: The research was carried out at the Jet Propulsion Laboratory, California 
##Institute of Technology, under a contract with the National Aeronautics and Space 
##Administration (80NM0018D0004) © 2025. California Institute of Technology. Government ##sponsorship acknowledged.

## -------------------------------- Functions ------------------------------- 

def Clip_Parent_Water_Mask(Parent_Image, Temp_Eco_File, Temp_Water_File, Temp_QC_File, Temp_Cloud_File):
    # Open the parent image and get its image specs
    Parent_Image = gdal.Open(Parent_Image)
    ref_x_min, ref_x_res, _, ref_y_max, _, ref_y_res = Parent_Image.GetGeoTransform()
    ref_width = Parent_Image.RasterXSize
    ref_height = Parent_Image.RasterYSize
    ref_x_max = ref_x_min + ref_x_res * ref_width
    ref_y_min = ref_y_max + ref_y_res * ref_height

    Eco_Aligned = np.zeros((ref_height, ref_width)) # Making duplicate zero Parent_images to keep image sizes the same
    Water_Aligned = np.zeros((ref_height, ref_width))
    QC_Aligned = np.zeros((ref_height, ref_width))
    Cloud_Aligned = np.zeros((ref_height, ref_width))

    Temp_Eco = gdal.Open(Temp_Eco_File) # Get current image values to match with the Parent WM
    temp_x_min, temp_x_res, _, temp_y_max, _, temp_y_res = Temp_Eco.GetGeoTransform()
    temp_width = Temp_Eco.RasterXSize
    temp_height = Temp_Eco.RasterYSize
    temp_x_max = temp_x_min + temp_x_res * temp_width
    temp_y_min = temp_y_max + temp_y_res * temp_height

    Temp_Water = gdal.Open(Temp_Water_File)
    Temp_QC = gdal.Open(Temp_QC_File)
    if os.path.exists(Temp_Cloud_File):
        Temp_Cloud = gdal.Open(Temp_Cloud_File)

    overlap_x_min = max(ref_x_min, temp_x_min) # Get the area of the overlapping segments between the temp eco file and the parent WM
    overlap_y_min = max(ref_y_min, temp_y_min)
    overlap_x_max = min(ref_x_max, temp_x_max)
    overlap_y_max = min(ref_y_max, temp_y_max)

    TL_x_parent = int((overlap_x_min - ref_x_min) / ref_x_res) # Calculate the top left and bottom right coordinates of the overlapped region for parent using parent resolution
    TL_y_parent = int((ref_y_max - overlap_y_max) / abs(ref_y_res))
    BR_x_parent = int((overlap_x_max - ref_x_min) / ref_x_res)
    BR_y_parent = int((ref_y_max - overlap_y_min) / abs(ref_y_res))

    cols_parent = BR_x_parent - TL_x_parent
    rows_parent = BR_y_parent - TL_y_parent

    
    if cols_parent <= 0 or rows_parent <= 0: # If statement to check if the temp scene is within the parent WM
        raise ValueError("Overlap Error")

    # Calculate the offset and number of pixels to read from the temp images
    TL_x_temp = int((overlap_x_min - temp_x_min) / temp_x_res) # Calculate the top left and bottom right coordinates of the overlapped region for temp images using temps resolution
    TL_y_temp = int((temp_y_max - overlap_y_max) / abs(temp_y_res))
    BR_x_temp = int((overlap_x_max - temp_x_min) / temp_x_res)
    BR_y_temp = int((temp_y_max - overlap_y_min) / abs(temp_y_res))

    cols_temp = BR_x_temp - TL_x_temp
    rows_temp = BR_y_temp - TL_y_temp

    if cols_temp <= 0 or rows_temp <= 0: # If statement to check if the temp scene is within the parent WM
        raise ValueError("Overlap Error")

    Eco_Clip = Temp_Eco.GetRasterBand(1).ReadAsArray(TL_x_temp, TL_y_temp, cols_temp, rows_temp).astype(float) # Reading the temp images from the overlapped segments. This avoids reading any part of the scene that is not needed
    Water_Clip = Temp_Water.GetRasterBand(1).ReadAsArray(TL_x_temp, TL_y_temp, cols_temp, rows_temp)           # This also helps with not taking too much memory
    QC_Clip = Temp_QC.GetRasterBand(1).ReadAsArray(TL_x_temp, TL_y_temp, cols_temp, rows_temp).astype(float) 

    if os.path.exists(Temp_Cloud_File):
        Cloud_Clip = Temp_Cloud.GetRasterBand(1).ReadAsArray(TL_x_temp, TL_y_temp, cols_temp, rows_temp)


    Eco_Aligned[TL_y_parent:BR_y_parent, TL_x_parent:BR_x_parent] = Eco_Clip[:rows_parent, :cols_parent] # Placing the Eco_Clip into the zeroed version of the temp file. This will be the "correct" position in an array
    Water_Aligned[TL_y_parent:BR_y_parent, TL_x_parent:BR_x_parent] = Water_Clip[:rows_parent, :cols_parent]
    QC_Aligned[TL_y_parent:BR_y_parent, TL_x_parent:BR_x_parent] = QC_Clip[:rows_parent, :cols_parent]

    if os.path.exists(Temp_Cloud_File):
        Cloud_Aligned[TL_y_parent:BR_y_parent, TL_x_parent:BR_x_parent] = Cloud_Clip[:rows_parent, :cols_parent]

    Parent_Aligned = Parent_Image.ReadAsArray()
    # Close the datasets
    Parent_Image = None
    Temp_Eco = None
    Temp_Water = None
    Temp_QC = None
    Temp_Cloud = None

    return Eco_Aligned.astype(float), Water_Aligned, QC_Aligned.astype(float), Cloud_Aligned, Parent_Aligned

def Best_Line_Eco(Line_Number, Ref_img, Og_Ref_img, Img_to_Clip,expand,step):
    expand = int(expand)
    
    Reference_Img = np.pad(Ref_img, ((expand, expand), (expand, expand)), mode='constant', constant_values=0)
    
    Image_to_Clip = np.pad(Img_to_Clip, ((expand, expand), (expand, expand)), mode='constant', constant_values=0)

    stats = measure.regionprops_table(Reference_Img,properties=('label','bbox','bbox_area'))
    
    Line_Box_Boundaries = DataFrame(stats)

    Line_Box_Boundaries = Line_Box_Boundaries.sort_values('bbox_area', ascending = False)

    Filter = (Line_Box_Boundaries['bbox_area'] > 1000).to_list()   # Filter the water mask bounding boxes to a 1000 pixels area (T or F)

    Binary_Filter = [1 if x else 0 for x in Filter] # Switched Filter from True or False to binary
  
    Labels = Line_Box_Boundaries['label'].to_list()                

    maxlines = Labels[Line_Number]

    if step == 0:

        Labels = np.multiply(Binary_Filter, Line_Box_Boundaries['label'].to_list())

        Mask = np.isin(Reference_Img, test_elements = Labels)
    else:

        Mask = np.isin(Reference_Img, test_elements = maxlines)

    maxlines = maxlines-1 

    if step == 0:

        # Setting up the extended boundary box for the ECOSTRESS line. 
        Pad_Col_S = int(abs(Line_Box_Boundaries['bbox-0'].min() - expand))
        Pad_Col_E = int(Line_Box_Boundaries['bbox-2'].max() + expand) 
        Pad_Row_S = int(abs(Line_Box_Boundaries['bbox-1'].min() - expand))
        Pad_Row_E = int(Line_Box_Boundaries['bbox-3'].max() + expand)

        Original_Shoreline = np.pad(Og_Ref_img, ((expand, expand), (expand, expand)), mode='constant', constant_values=0)
        
        Mask = Mask[Pad_Col_S:Pad_Col_E, Pad_Row_S:Pad_Row_E]
        
        Clipped_Shoreline = Original_Shoreline[Pad_Col_S:Pad_Col_E, Pad_Row_S:Pad_Row_E]

        Clipped_Reference = Clipped_Shoreline * Mask

        Clipped_ITC = Image_to_Clip[Pad_Col_S:Pad_Col_E, Pad_Row_S:Pad_Row_E]
    
        Clip_Values = (Pad_Col_S,Pad_Col_E, Pad_Row_S,Pad_Row_E)

    if step == 1:
        # Setting up the extended boundary box for the ECOSTRESS line. 
        Pad_Col_S = int(abs(Line_Box_Boundaries.at[maxlines,'bbox-0'] - expand))
        Pad_Col_E = int(Line_Box_Boundaries.at[maxlines,'bbox-2'] + expand) 
        Pad_Row_S = int(abs(Line_Box_Boundaries.at[maxlines,'bbox-1'] - expand))
        Pad_Row_E = int(Line_Box_Boundaries.at[maxlines,'bbox-3'] + expand)

        Clipped_Reference = np.clip(Mask[Pad_Col_S:Pad_Col_E,Pad_Row_S:Pad_Row_E],a_min=0, a_max=1)

        Clipped_ITC = np.clip(Image_to_Clip[Pad_Col_S:Pad_Col_E,Pad_Row_S:Pad_Row_E],a_min=0, a_max=1) # ITC is Image to Clip

        Clip_Values = (Pad_Col_S,Pad_Col_E, Pad_Row_S,Pad_Row_E)

    Col_length = Line_Box_Boundaries.at[maxlines,'bbox-1']

    Row_length = Line_Box_Boundaries.at[maxlines,'bbox-3']

    return Clipped_Reference, Clipped_ITC, Clip_Values,Col_length, Row_length,maxlines

def Sorted_Longest_Lines(img):
    Labeled_Img = measure.label(img, connectivity = 2)
    
    Labeled_Table = measure.regionprops_table(Labeled_Img, properties=('label','bbox','bbox_area'))

    Labeled_Table = DataFrame(Labeled_Table)

    Line_Lengths = []

    for i in range(1, Labeled_Table.shape[0]+1):
        count = np.count_nonzero(Labeled_Img == i)
        Line_Lengths.append(count)

    Labeled_Table.insert(6, "Line_Length", Line_Lengths, True)

    Sorted_Table = Labeled_Table.sort_values('Line_Length', ascending = False)

    return Sorted_Table, Labeled_Img

def End_Points(img):

    Skeleton_ECOSTRESS = (morphology.skeletonize(img)).astype(np.uint8)
    
    directions = [
        np.array([[1,0,0],[0,1,0],[0,0,0]]).astype(np.uint8),
        np.array([[0,1,0],[0,1,0],[0,0,0]]).astype(np.uint8),
        np.array([[0,0,1],[0,1,0],[0,0,0]]).astype(np.uint8),
        np.array([[0,0,0],[1,1,0],[0,0,0]]).astype(np.uint8),
        np.array([[0,0,0],[0,1,1],[0,0,0]]).astype(np.uint8),
        np.array([[0,0,0],[0,1,0],[1,0,0]]).astype(np.uint8),
        np.array([[0,0,0],[0,1,0],[0,1,0]]).astype(np.uint8),
        np.array([[0,0,0],[0,1,0],[0,0,1]]).astype(np.uint8)
    ]

    Combined_Results = sum(binary_hit_or_miss(Skeleton_ECOSTRESS, structure1 = EndPoint_Orientation).astype(int) for EndPoint_Orientation in directions)

    endpoint_locations = np.where(Combined_Results == 1)

    EndP_Locations = list(zip(endpoint_locations[0],endpoint_locations[1])) # Row (Y), Column order (X)

    # Some lines are circular and do not have endpoints this section provides a work around to no end points 
    if len(EndP_Locations) == 0:
        Every_ones_Location = np.where(Skeleton_ECOSTRESS == 1)

        Total_Location = list(zip(Every_ones_Location[0],Every_ones_Location[1]))

        EndP_Locations = Total_Location[0]

    return Skeleton_ECOSTRESS, EndP_Locations

def MatchingOverlay(Eco,Water, expand, endpoints):

    if type(endpoints) is list:
        yend_Eco = endpoints[0][0]
        xend_Eco = endpoints[0][1]
    if type(endpoints) is tuple:
        yend_Eco = endpoints[0]
        xend_Eco = endpoints[1]

    Water_Coordinates = np.argwhere(Water == 1)

    xs = []
    ys = []
    
    for i in range(0, len(Water_Coordinates)):
        ys.append(Water_Coordinates[i][0])
        xs.append(Water_Coordinates[i][1])

    # Calculate binary list of all endpoints within 100 pixels

    positions = (xs <= (xend_Eco + expand)) & (xs >= (xend_Eco - expand)) & \
                (ys <= (yend_Eco + expand)) & (ys >= (yend_Eco - expand))

    # Convert binary list to shift coordinates
    Shift_Coordinates = np.column_stack((positions * (xs*positions-xend_Eco), positions * (ys*positions-yend_Eco)))

    # Find rows with exactly two zeros
    mask = np.count_nonzero(Shift_Coordinates, axis=1) == Shift_Coordinates.shape[1] - 2

    filtered_arr = Shift_Coordinates[~mask]

    filtered_arr = np.vstack((filtered_arr,[0,0]))

    [X_Shift, Y_Shift, Eco_Ratio, _, Overlapped_Twos] = Matching_Shift(filtered_arr,Eco,Water)

    return X_Shift, Y_Shift, Eco_Ratio, Overlapped_Twos

def Matching_Shift(Shift_Coordinates, Eco, Water):
    # Initialize variables
    Best_Position = 0
    Best_Index = 0
    Eco_Ratio = 0
    Connected_Twos = []
    
    # Iterate through Shift_Coordinates
    for i in range(len(Shift_Coordinates)):

        Shifted_Image = np.roll(Eco, (Shift_Coordinates[i]),axis = (1,0))

        New_Combined = Shifted_Image + Water

        # Measurement of connected components (Lengths)

        New_Twos = np.count_nonzero(New_Combined == 2)

        if New_Twos > Best_Position:                # New positions
            Best_Position = New_Twos
            Best_Index = i
            Eco_Ratio = New_Twos/ np.count_nonzero(Eco == 1)
            Connected_Twos = New_Combined         

    if Best_Position == 0: # Incase the initial position is best 
        X_Shift = 0
        Y_Shift = 0
        Eco_Ratio = 0
        Connected_Twos = Eco + Water
    else:
        Y_Shift = Shift_Coordinates[Best_Index, 1]
        X_Shift = Shift_Coordinates[Best_Index, 0]

    return X_Shift, Y_Shift, Eco_Ratio, Connected_Twos, Best_Position

def Shift_in_Progress(Eco_File_Temp, Folder, New_Folder, Parent_WM_File, Use_Personalized_WaterMask, Use_Cloud_Mask):

    ## ----------------- File Setup -----------------
    doy = r"doy\d{13}"
    doy_num = (re.search(doy,Eco_File_Temp)).group()

    QC_File_temp = Eco_File_Temp.replace('_LST_', '_QC_')

    EcoFile = os.path.join(Folder, Eco_File_Temp)    
    QCFile = os.path.join(Folder, QC_File_temp)

    lst_check = os.path.exists(EcoFile)
    qc_check = os.path.exists(QCFile) 

    if lst_check == qc_check:
        
        WaterFile = None

        # ----------------- Water and Cloud File Identification ----------------- 
        if "_L2_" in Eco_File_Temp:
            Water_File_temp = Eco_File_Temp.replace('_LST_', '_water_mask_')
            Cloud_File_temp = Eco_File_Temp.replace('_LST_', '_cloud_mask_')

            WaterFile = os.path.join(Folder, Water_File_temp)
            CloudFile = os.path.join(Folder, Cloud_File_temp)

            if os.path.exists(WaterFile):
                pass
            
            if Use_Personalized_WaterMask == True:  
                Water = gdal.Open(WaterFile).ReadAsArray()                # Individual Water Mask
                QC = gdal.Open(QCFile).ReadAsArray().astype(float)
                Eco = gdal.Open(EcoFile).ReadAsArray().astype(float)
                if os.path.exists(CloudFile):
                    Cloud = gdal.Open(CloudFile).ReadAsArray()    
                
            if Use_Personalized_WaterMask == False:
                Parent_WM_File = os.path.join(Folder, Parent_WM_File)
                Eco, _, QC, Cloud, Water = Clip_Parent_Water_Mask(Parent_WM_File, EcoFile, WaterFile,QCFile,CloudFile)

        elif "_L2T_" in Eco_File_Temp:

            Water_File_temp = Eco_File_Temp.replace('_LST_', '_water_')
            Cloud_File_temp = Eco_File_Temp.replace('_LST_', '_cloud_')

            WaterFile = os.path.join(Folder, Water_File_temp)
            CloudFile = os.path.join(Folder, Cloud_File_temp)

            if Use_Personalized_WaterMask == True:  
                Water = gdal.Open(WaterFile).ReadAsArray()                # Individual Water Mask
                QC = gdal.Open(QCFile).ReadAsArray().astype(float)
                Eco = gdal.Open(EcoFile).ReadAsArray().astype(float)
                if os.path.exists(CloudFile):
                    Cloud = gdal.Open(CloudFile).ReadAsArray()    
               
            if Use_Personalized_WaterMask == False:
                Parent_WM_File = os.path.join(Folder, Parent_WM_File)
                Eco, _, QC, Cloud,Water = Clip_Parent_Water_Mask(Parent_WM_File, EcoFile, WaterFile,QCFile,CloudFile)
            
        ## ----------------- Shore Availability Check -----------------
        if not np.array_equal(Water, np.zeros(Water.shape)):

            Eco[Eco == 0] = np.nan

            ## -------------------------Determining Acceptable Edges from QC File -------------------------

            Accetable_Values_Area = ((QC != -99999) & (QC != 65535) & (QC != 0)).astype(np.uint8)

            Erase = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)).astype(np.uint8)

            Eroded_QC = cv2.erode(src = Accetable_Values_Area, kernel = Erase).astype(np.uint8)

            Eroded_QC =  Eroded_QC.astype(np.uint8) # Eroded_QC will be used to remove the water mask border edges for large swath images. 

            ## --------------------------- Removing Image Edges Detected in Water Mask ---------------------------
            Water = np.nan_to_num(Water, nan=0).astype(np.uint8)
            Water[Water == 255] = 0  # 0 over NaN due to desired edge detection

            Inverse_Water = cv2.bitwise_not((Water * 255)).astype(np.uint8)

            Inverse_Water_255 = (Inverse_Water * 255).astype(np.uint8)

            Inverse_dist_transform_water = cv2.distanceTransform(src = Inverse_Water_255, distanceType =  cv2.DIST_L2, maskSize = 5)                     

            Distance_to_Edge = Inverse_dist_transform_water == 1

            Inverse_Accetable_Values_Area = (Distance_to_Edge).astype(np.uint8) * Eroded_QC # Pixels with values of 1 are the shoreline of the water mask.
                                                                                            # Applying the Eroded_QC removes the image edges that are not needed for calculation

            del Water, Inverse_Water, Inverse_Water_255, Inverse_dist_transform_water, Distance_to_Edge
            gc.collect()
            ## ------------------------- Applying Cloud Mask to LST and Water Mask (if possible) -------------------------
            if not np.array_equal(Inverse_Accetable_Values_Area, np.zeros(Inverse_Accetable_Values_Area.shape)):

                ##  ------------------------- Cloud Mask to LST -------------------------
                if Use_Cloud_Mask == True:
                    
                    Cloud[Cloud == np.nan] = 0
                    Cloud = Cloud.astype(np.uint8)
                    Cloud_255 = (Cloud * 255).astype(np.uint8)

                    dist_transform = cv2.distanceTransform(Cloud_255, cv2.DIST_L2, 5)

                    dist = 5

                    _, Small_Cloud_Edge_Dilate = cv2.threshold(dist_transform,dist,255, cv2.THRESH_BINARY) # A radius of 5 ensures edges based on the clouds are still used with edge detection. Set values to 255 for bitwise inversion.

                    Small_Cloud_Edge_Dilate = cv2.bitwise_not(Small_Cloud_Edge_Dilate.astype(np.uint8)) 

                    Eco_Masked = Eco * Small_Cloud_Edge_Dilate

                    Eco_Masked[Eco_Masked == 0] = np.nan  # cleaning up Eco_Masked 0's

                    ## ------------------------- Cloud Mask on Water Mask  -------------------------
                    dist2 = 100

                    _, Cloud_Edge_Eroded = cv2.threshold(dist_transform,dist2,255, cv2.THRESH_BINARY)

                    Cloud_Edge_Eroded = cv2.bitwise_not(Cloud_Edge_Eroded.astype(np.uint8)) / 255

                    Shoreline_without_Image_Boundary = Inverse_Accetable_Values_Area * Cloud_Edge_Eroded

                    Shoreline_without_Image_Boundary[Shoreline_without_Image_Boundary == 0] = 0  # cleaning up Eco_Masked 0's
                else:

                    Eco_Masked = Eco

                    Eco_Masked[Eco_Masked == 0] = np.nan  # cleaning up Eco_Masked 0's
                    
                    Shoreline_without_Image_Boundary = Inverse_Accetable_Values_Area

                ## --------------------------- Identifying the Longest Shoreline ---------------------------

                Connecting_Shoreline = morphology.binary_dilation(Shoreline_without_Image_Boundary).astype(np.uint8) # Increases thickness of lines to connect any disconnected lines

                _,Mapped_Lines = cv2.connectedComponents(Connecting_Shoreline, connectivity = 8)
                
                [Clip_Water, Clip_Eco,_, _,_,_] = Best_Line_Eco(0, Mapped_Lines,Shoreline_without_Image_Boundary, Eco_Masked,100,0) # Clips both Water and ECOSTRESS image to the longest Shoreline

                _,Mapped_Lines2,stats,_ = cv2.connectedComponentsWithStats(Clip_Water.astype(np.uint8), connectivity = 8)
                
                Shore_lengths = stats[: ,cv2.CC_STAT_AREA]

                sorted_indecies = np.argsort(-Shore_lengths)

                Blank = np.array([[0,0],[0,0]]).astype(np.uint8)  # This "Blank" variable is needed to avoid an optional route in the custom Best_Line_Eco formula

                Main_Matching_Stats = []

                Clip_Eco[Clip_Eco == 0] = np.nan

                del Connecting_Shoreline, Mapped_Lines, Shore_lengths
                gc.collect()

                ## --------------------------- ECOSTRESS Edge Area Clip ---------------------------             

                Water_Buffer = morphology.isotropic_dilation(Clip_Water,radius = 100)   # Applied a 100 radius buffer to Clip_Water representing the area a possible shift is possible

                Clip_Eco[~Water_Buffer] = np.nan    # Applies mask from Water Buffer (removes any non accetable shift option)

                ## --------------------------- ECOSTRESS Edge Detection and Line Selections ---------------------------
                
                Initial_Canny =  feature.canny(Clip_Eco,sigma=2.5)  # This value can be changed for certain LST images. Sigma = 2 is a good balance between

                High_Blur_Canny =  feature.canny(Clip_Eco,sigma=6)  # Higher order edge used to simplify image-to-match  

                Labeled_Edges = measure.label(Initial_Canny, connectivity = 2) # Gives an number to each binary line

                Overlapping_Lines = Labeled_Edges * High_Blur_Canny   # Keeps lines that overlap between the heavily blurred lines and the intial canny. Heavy blurred lines have reduce shoreline accuracy

                Distinct_Line_Nums = np.unique(Overlapping_Lines)        # Isolates values that overlapped in Overlapiping_LinesHigh_Edge_Clip

                Distinct_Line_Nums = list(Distinct_Line_Nums[1:])       # Removes 0 from the unique values

                Distinct_Line_Selection = np.isin(Labeled_Edges, test_elements = Distinct_Line_Nums) # Uses the overlapped indexes to filter the clipped canny edges

                del Water_Buffer, Clip_Eco,Initial_Canny, High_Blur_Canny, Labeled_Edges, Distinct_Line_Nums, Overlapping_Lines
                gc.collect()

                ## --------------------------- Selection of Top ECOSTRESS Edges  --------------------------
                
                for Labeled_line in sorted_indecies:
                    if Labeled_line == 0:
                        continue

                    Isolated_Shore = (Mapped_Lines2 == Labeled_line).astype(np.uint8)

                    [Indv_Water_Line, Indv_Eco_Line,_, _,_,_] = Best_Line_Eco(0, Isolated_Shore,Blank, Distinct_Line_Selection ,100,1)

                    Sorted_Table, Labeled_ECOSTRESS_Image = Sorted_Longest_Lines(Indv_Eco_Line)

                    Top_2 = Sorted_Table.iloc[:2, -1].values

                    if Top_2.size < 2:
                        continue

                    First_Row = Top_2[0]
                    Second_Row = Top_2[1]

                    # Sorted Line selections will get shorter based on the longest identified area containing the longest line. Doing so reduces computation times
                    if First_Row >= (Second_Row * 2) and First_Row >=50000:
                        Index_values = ((Sorted_Table.head(5))['label']).to_list()    # Store the labels of the top 5 longest lines
                    else:
                        Index_values = ((Sorted_Table.head(100))['label']).to_list()    # Store the labels of the top 100 longest lines

                    Sorted_Lines_Map = np.isin(Labeled_ECOSTRESS_Image, test_elements = Index_values) # Returns a binary image

                    Newly_Labeled = measure.label(Sorted_Lines_Map, connectivity = 2) # Labeled again to begin a new analysis from the binary Sorted_Lines_Map

                    Temp_Matching_Stats = []
                    ## --------------------------- Acquiring Eligable Shifts --------------------------- 

                    for i in range(0, len(Index_values)):
                        [Eco_Line, Water_Line,_, Col_length, Row_length,Actual_Index] = Best_Line_Eco(i,Newly_Labeled,Blank,Indv_Water_Line,100,1)

                        Skeleton_image, Endpoints = End_Points(Eco_Line) 
                        
                        if len(Endpoints) <=6: 

                            Skeleton_image = morphology.isotropic_dilation(Skeleton_image, radius = 1)

                            [Col_Shift,Row_Shift, Ratio,_] = MatchingOverlay(Skeleton_image, Water_Line, 100 , Endpoints)

                            Temp_Matching_Stats.append([Col_Shift, Row_Shift, (Ratio*Ratio)*Col_length*Row_length,(Actual_Index+1)]) # The "+ 1" was needed to correct the line's index value to its labeled value
                        else: 
                            pass
                    
                ## --------------------------- Shift Check for Combined Top Lines --------------------------- 
                    Temp_Sorted_Matching_Stats = sorted(Temp_Matching_Stats, key=lambda x:x[2], reverse=True)

                    Temp_X_Shifts = [i[0] for i in Temp_Sorted_Matching_Stats]
                    Temp_Y_Shifts = [i[1] for i in Temp_Sorted_Matching_Stats]
                    Temp_Best_Indexes = [i[3] for i in Temp_Sorted_Matching_Stats]

                    Temp_Shift_Coordinates = np.column_stack((Temp_X_Shifts,Temp_Y_Shifts))

                    Temp_Top_Lines = np.isin(Newly_Labeled, test_elements = Temp_Best_Indexes)

                    Temp_Buffered_Lines = morphology.isotropic_dilation(Temp_Top_Lines, radius = 1)

                    [Temp_X_Shift, Temp_Y_Shift, _ ,_, _] = Matching_Shift(Temp_Shift_Coordinates,Temp_Buffered_Lines,Indv_Water_Line)

                    Main_Matching_Stats.append([Temp_X_Shift,Temp_Y_Shift])  

                # --------------------------- Testing X and Y Shifts ---------------------------
                Main_X_Shifts = [i[0] for i in Main_Matching_Stats]
                Main_Y_Shifts = [i[1] for i in Main_Matching_Stats]

                Main_Buffered_Lines = morphology.isotropic_dilation(Distinct_Line_Selection, radius = 1)

                Main_Shift_Coordinates = np.column_stack((Main_X_Shifts,Main_Y_Shifts))

                [Main_X_Shift, Main_Y_Shift, _,_, _] = Matching_Shift(Main_Shift_Coordinates,Main_Buffered_Lines,Clip_Water)

                print('Shift correction is being run for ', Eco_File_Temp)
                print('Final Shift')
                print(f'x-shift is {Main_X_Shift} \n')
                print(f'y-shift is {Main_Y_Shift} \n') 

                ## ------------------ Applying the shift ------------------

                Eco_Main = gdal.Open(EcoFile)

                driver = gdal.GetDriverByName('GTiff')

                New_Name = os.path.join(New_Folder, 'Shifted_' + Eco_File_Temp) 

                New_File = driver.CreateCopy(New_Name, Eco_Main, strict = 0)
                New_File = None

                Shift_File = gdal.Open(New_Name,gdal.GA_Update)

                File_Variables = Shift_File.GetGeoTransform()

                x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = File_Variables

                shift_x = Main_X_Shift * x_res
                shift_y = Main_Y_Shift * y_res

                gt_update = (x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res)

                Shift_File.SetGeoTransform(gt_update)

                Shift_File.FlushCache()

                ## --------------------------- Closing gdal open rasters ---------------------------
                Eco = None
                Water = None
                QC = None
                Shift_File = None

                return doy_num, Eco_File_Temp, int(Main_X_Shift), int(Main_Y_Shift), 'Complete'
            else:
                return doy_num, Eco_File_Temp, np.nan, np.nan, 'Error - No Shoreline'
        else:
            return doy_num, Eco_File_Temp, np.nan, np.nan, 'Error - No Shoreline'
    else:
        return doy_num, Eco_File_Temp, np.nan, np.nan, 'Error - No QC or LST'

def main():
    ## ---------------------------------- Reading and Creating Folders ----------------------------------
    Start_Code = time.time()
    # Must have ' / ' at the end of your folder path 

    Folder = r'C:/Users/User/Folder/Here/' # Must have "/" at the end
                   
    New_Folder = Folder + "Shifted_Folder/"             # New folder where shifted copies will be created

    if not os.path.exists(New_Folder):                      
        os.mkdir(New_Folder)

    ## ---------------------------------- Set Variables ---------------------------------- 
    Use_Cloud_Mask = True                                                       # Set to True by default to use the cloud mask to clip LST images
    Use_Personalized_WaterMask = True                                            # Set to True by default to use personalized water masks for each scene.

                                                                                # If Use_Personalized_WaterMask is set to False, 
    Main_WM_File = 'None'    # the code will use Main_WM_File as the water mask for all scenes
                                                                                # Specify the watermask file name and set the Use_Personalized_WaterMask to False in order to use this option

    ## ---------------------------------- LST Flag Comparison (Tentative) ---------------------------------- 
    # Current file contains flags between the following dates ----> July 9th, 2018 to November 29th, 2024
    Flags_of_Interest = [ "Poor", "Suspect"]
    
    All_SP_DOYS = []

    with open('SP_Geo_Flags.txt','r') as file:     # Location to the Geolocation Flags file
        for line in file:
            if any(GeoFlag in line for GeoFlag in Flags_of_Interest):
                Standard_Date = (line.strip())[26:34]
                ECOSTRESS_Doy_Conversion = datetime.strptime(Standard_Date, "%Y%m%d").strftime("%Y%j")
                HMS_Time = (line.strip())[35:41]
                Full_Doy = "doy" + ECOSTRESS_Doy_Conversion + HMS_Time
                All_SP_DOYS.append(Full_Doy)

    LST_filtered_files = []
    
    for file in os.listdir(Folder):
        if file.endswith('.tif') and '_LST_' in file:
            doy = re.search(r'doy(\d{13})', file).group()
            if doy in All_SP_DOYS:
                LST_filtered_files.append(file)

    ## ---------------------------------- AppEEARS Statistics CSV ----------------------------------
    # Uncomment this section when AppEEARS provides Geoflags for Swath AND Tiled ECOSTRESS scenes. If only Swath scenes are available, this section can be used.
    #
    # LST_filtered_files = []

    # for file in os.listdir(Folder):
    #     if file.endswith('.csv'):
    #         with open(os.path.join(Folder,file), 'r') as file:
    #             reader = csv.reader(file)
    #             headers = next(reader)
    #             if 'File Name' in headers and 'Geolocation Accuracy QA' in headers:
    #                 FileName = headers.index('File Name')
    #                 GeoFlag = headers.index('Geolocation Accuracy QA')
    #                 for row in reader:
    #                     if row[GeoFlag] == 'Bad' or row[GeoFlag] == 'Suspect':
    #                         LST_filtered_files.append(row[FileName])

    ## ---------------------------------- Multi-Processing ---------------------------------- 

    Hold_Futures = []  
    num_workers = os.cpu_count() # - # of cores to free up                            # Number of logical cores. This will be the same as "None" from the concurrent.futures.ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file in LST_filtered_files[::-1]:
                try:
                    future = executor.submit(Shift_in_Progress, file, Folder, New_Folder, Main_WM_File, Use_Personalized_WaterMask, Use_Cloud_Mask)
                    Hold_Futures.append(future)
                except (ValueError, FileNotFoundError, Exception):  # This except was made for files that do not overlap with the parent water mask 
                    pass            
    # Ensure all futures are completed before measuring the final time
    Results, _ = concurrent.futures.wait(Hold_Futures, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)

    ## ---------------------------------- CSV script ---------------------------------- 
    
    CSV_File =[]

    for future in Results:              # Incase of errors with future.result(), this try and except will skip and only keep the 
        try:                            # successful results. Examples of errors are when files are not found due to issues with AppEEARS requests.
            future.result()
            CSV_File.append(future.result())
        except (ValueError, FileNotFoundError, Exception):
            pass

    Results_Path = os.path.join(New_Folder, 'ECOSTRESS_Correction_results.csv')

    with open(Results_Path, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['doy','File Name','X-Shift','Y-Shift', 'Result'])
        writer.writerows(CSV_File)
    
    print(f'Final Run Time: {time.time() - Start_Code}')

## ----------------------------- Running Code  -----------------------------  
if __name__ == '__main__':
    freeze_support()
    main()

