import pydicom # DICOM
import os
import fnmatch # Filter
import numpy as np
from sklearn.decomposition import FastICA # ICA
import matplotlib.pyplot as plt
import pandas as pd

from skimage.segmentation import flood, flood_fill
import nibabel
import time



def getImageData(dir, my_path):
    '''Returns the 4D array of a PET-image.
    '''

    os.chdir(dir)

    file_list=fnmatch.filter(os.listdir(my_path), '*.dcm')

    # Find dimension of the image from the first dcm file
    tmp_file=pydicom.dcmread(my_path+'/'+file_list[0])
    x_dim=tmp_file[0x0028, 0x0010].value # Rows
    y_dim=tmp_file[0x0028, 0x0011].value # Columns
    z_dim=tmp_file[0x0054, 0x0081].value # Number of Slices [0x0054, 0x0081]
    frame_dim=tmp_file[0x0054, 0x0101].value # Number of Time Slices

    # Check dimension and file numbers
    if(z_dim*frame_dim != len(file_list)):
        print('Dimension does not match!')

    # Allocate memory
    image_data=np.empty([x_dim,y_dim,z_dim,frame_dim])

    # Read all image files
    for k in range(len(file_list)):
        if (k%1000)==0:
            print(k)
        
        tmp_file=pydicom.dcmread(my_path+'/'+file_list[k])

        image_index=tmp_file[0x0054, 0x1330].value # Image Index
        
        tmp_image_data=tmp_file.pixel_array*tmp_file[0x0028, 0x1053].value+tmp_file[0x0028, 0x1052].value
        
        # Find z index
        z_ind=image_index % z_dim

        if z_ind==0:
            z_ind=z_dim
            
        z_ind=z_ind-1
        
        # Find frame number
        frame_ind=np.floor((image_index-1)/z_dim).astype(int)
        
        # Store data
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                image_data[x_ind,y_ind,z_ind,frame_ind]=tmp_image_data[x_ind,y_ind]

    sum(sum(sum(sum(image_data))))

    return image_data


def modifyImageData_3D(image_data_3d, clear_background=False, tol=0.025, verbose=1):
    '''
    Scale and modify the imagedata a bit for easier handling.

    Clear background = if True, clears the background within some tolerance.

    Tolerance = tolerance for above operation in percents, values in [0, 1].
    '''

    # Take max and min of original voxel values.
    maxx =np.max(image_data_3d)
    minn =np.min(image_data_3d)
    if verbose==1:
        print("Original image_data limit values:", maxx, minn)


    image_data_3d_new = image_data_3d + (-1*minn)

    # for i in range(np.shape(image_data_3d)[0]):
        # for j in range(np.shape(image_data_3d)[1]):
            # for k in range(np.shape(image_data_3d)[2]):
                # image_data_3d[i][j][k] = image_data_3d[i][j][k] + (-1*minn)


    #70, 0, 70
    if clear_background == True:

        image_data_3d_new = flood_fill(image_data_3d_new, (70, 0, 70), 0, tolerance=tol*maxx)

        # for i in range(np.shape(image_data_3d)[0]):
        #     for j in range(np.shape(image_data_3d)[1]):
        #         for k in range(np.shape(image_data_3d)[2]):
        #             if image_data_3d[i][j][k] < maxx*tol:
        #                 image_data_3d[i][j][k] = 0
                        

    maxx =np.max(image_data_3d_new)
    minn =np.min(image_data_3d_new)
    if verbose==1:
        print("Modified and scaled image_data limit values:", maxx, minn)

    return image_data_3d_new



def hillClimbing_3d(image_data, seed_point, scope):
    '''
    Based on hillClimbing2(). Idea is the same but in 3D. Used for dicom and nifti files to get hotspots.

    image_data is 3D array of pixel values
    scope is the area we use for our local max search (tuple of 3 integers)
    seed point is the starting point of local max search


    1. Take seed point and check max in scope*scope*scope around the point (seed_point is the middle point). scope == 2 means we check 5*5*5 area. The initial seed point and two voxels around it in every direction.
    2. Check if multiple local maxes.
    3. Save local max coordinates (in local context). If multiple, take the first occurrence
    4. Calculate global coordinates from local coordinates and the original scope.
    5. Check if seed point is the local max. If so, return (no better directions). If not, continue recursion with the new seed point as input.

    '''

    # Find the max value in local seed_value enviroment
    # Seed point is in the middle

    x_scope = scope
    y_scope = scope
    z_scope = scope


    if(seed_point[0]-scope < 0):

        x_offset = seed_point[0]-scope
        x_scope = x_scope + x_offset
    
    if(seed_point[1]-scope < 0):

        y_offset = seed_point[1]-scope
        y_scope = y_scope + y_offset
    
    if(seed_point[2]-scope < 0):

        z_offset = seed_point[2]-scope
        z_scope = z_scope + z_offset
          
        

    max_value = np.max(                     image_data[seed_point[0] - x_scope:seed_point[0] + scope+1, seed_point[1] - y_scope:seed_point[1] + scope+1, seed_point[2] - z_scope:seed_point[2] + scope+1])

    current_hill_local_multiples = np.where(image_data[seed_point[0] - x_scope:seed_point[0] + scope+1, seed_point[1] - y_scope:seed_point[1] + scope+1, seed_point[2] - z_scope:seed_point[2] + scope+1] == max_value)



    # Current highest point coordinates in the local enviroment (scope x scope)
    # TODO: name better
    # If multiple, takes the first occurence
    # Initialize the parameter that stores the current hill (biggest value in the local enviroment, not in the already scoped area)
    current_hill_local = [0, 0, 0]

    # Amount of points that are equal to the local maximum. If more than one, decide which point to take

    # Check if multiple max_values, if yes, take the first occurrence
    # TODO: don't take the first, take the farthest or something else. Implement if time
    #
    # np.shape(current_hill_local_multiples[0])[0] > 1 --> if multiples array contains more than one element, take first value
    if(np.shape(current_hill_local_multiples)[1] >= 2):
        for i in range(scope):
            for j in range(scope):
                for k in range(scope):
                    if(image_data[seed_point[0]-scope + i][seed_point[1]-scope + j][seed_point[2]-scope + k] == max_value):
                        # print(i, j, k)
                        # Max coordinates in LOCAL CONTEXT
                        # Set the current hill local here in DUPLICATES case
                        current_hill_local[0] = i
                        current_hill_local[1] = j
                        current_hill_local[2] = k
                        break
                    else:
                        continue
                break
    

    # Set the current hill local here in NO DUPLICATES case
    else:
        # current_hill_local = (current_hill_local_multiples[0], current_hill_local_multiples[1], current_hill_local_multiples[2])
        current_hill_local[0] = current_hill_local_multiples[0][0]
        current_hill_local[1] = current_hill_local_multiples[1][0]
        current_hill_local[2] = current_hill_local_multiples[2][0]


    # Assign the GLOBAL LOCATION of current hill to the variable
    # We sum the (seed_point indices - scope) to local scope indices to get the global coordinates of the current hill

    current_hill_x = seed_point[0]-x_scope + current_hill_local[0]
    current_hill_y = seed_point[1]-y_scope + current_hill_local[1]
    current_hill_z = seed_point[2]-z_scope + current_hill_local[2]


    # Check if seed_point is the local max. If yes, return. If not, continue.
    if(image_data[seed_point[0]][seed_point[1]][seed_point[2]] == max_value):
        #print(max_value, current_hill[0], current_hill[1])
        return max_value, current_hill_x, current_hill_y, current_hill_z

    # New seed point for recursion
    temp_tuple = (current_hill_x, current_hill_y, current_hill_z)

    return hillClimbing_3d(image_data, temp_tuple, scope)


def findHotspots_3d(image_data_3d, grid, scope, threshold=0.0, as_tuple=0, verbose=0):
    '''
    Returns a 3xN matrix of hotspots, where 
    
        [i,0] represents i-th x-component,

        [i,1] represents i-th y-component,

        [i,2] represents i-th z-component,

    grid = how loose you want the search grid to be. Bigger value may mean some hotspots are missed.

    scope = passed to hillClimbing_3d. How big is the search radius of new higher value.

    threshold = drop all hotspots under this threshold

    as_tuple = 1 if you want to return the hotspot as a tuple

    verbose = 1 for some info, 10 for all the info.
    '''

    #Initialize arrays of hotspots, each index i is the i-th hotspot coordinates in each array
    #Row
    hotspots_x = np.zeros(1, dtype=int)
    #Column
    hotspots_y = np.zeros(1, dtype=int)
    #Depth
    hotspots_z = np.zeros(1, dtype=int)


    # List of all voxels with value larger than 5% of max
    # TODO: make it input variable if it seems like a smart move

    maxx = np.max(image_data_3d)

    if verbose != 0:
        print("")
        print("Total voxel amount:", np.size(image_data_3d))
        print("Lazy search grid size =", grid, "(", grid*grid*grid, ")")
        print("Search radius around the seed point =", scope, "(", 2*scope+1, ")")
        print("")


    # Main loop
    # EVERY positive voxel is used as seed for hillclimbing
    over = 0
    under = 0
    count = 0

    # Create more efficient way. Split the matrix in slices and take the max from each and use that as a seed point for hillClimbing.
    for i in range((int(np.ceil(image_data_3d.shape[0] / grid)))):
        if verbose > 15:
            print(i)
            if i == 1 and verbose > 20:
                print("JUST ONE PLANE DONE!!")
                break
        for j in range((int(np.ceil(image_data_3d.shape[1] / grid)))):
            for k in range((int(np.ceil(image_data_3d.shape[2] / grid)))):

                
                local_maxx = np.max(image_data_3d[i*grid : (i+1)*grid, j*grid : (j+1)*grid, k*grid : (k+1)*grid])

                # Returns in max LOCAL SLICE CONTEXT
                temp_seed = np.where(image_data_3d[i*grid : (i+1)*grid, j*grid : (j+1)*grid, k*grid : (k+1)*grid] == local_maxx)

                seed = (temp_seed[0][0]+(i*grid), temp_seed[1][0]+(j*grid), temp_seed[2][0]+(k*grid))

                # Correct?
                # Check if the current seed is over threshold. If not, skip.
                if image_data_3d[seed[0]][seed[1]][seed[2]] <= threshold * maxx:
                    if verbose > 10:
                        under = under + 1
                    continue
                else:
                    if verbose > 10:
                        over = over + 1
                
                # Scope passed
                hotspot_result = hillClimbing_3d(image_data_3d, seed, scope)
                count = count + 1

                hill = (hotspot_result[1], hotspot_result[2], hotspot_result[3])

                # if verbose > 10:
                #     if (image_data_3d[hill[0]][hill[1]][hill[2]] > threshold * maxx):
                #         # print("OVER THRESHOLD: ", i, j, k)
                #         over = over + 1
                #     else:
                #         # print("UNDER THRESHOLD: ", i, j, k)
                #         under = under + 1

                #print(hotspot_result[0])

                # Secondary loop to check if the current hotspot (hill) is already saved in the list
                for l in range(np.shape(hotspots_x)[0]):

                    # Temporal variable, the l-th item in hotspot list. To make code more readable
                    temp_tuple = (hotspots_x[l], hotspots_y[l], hotspots_z[l])

                    # If current hotspot (hill) is already present in the hotspot list, break the loop and do nothing since we don't want duplicates
                    if(hill == temp_tuple):
                        break
                    
                    # If the current hotspot (hill) is not equal to the last index of hotspot list we can add current hotspot (hill) to the list.
                    # This is because the loop would have been broken already if there was a duplicate.
                    # Also check if the hotspots value is over threshold value. Used to filter out useless background noise hotspots.
                    if (image_data_3d[hill[0]][hill[1]][hill[2]] > threshold * maxx):
                        if(hill != temp_tuple and l == np.shape(hotspots_x)[0]-1):

                            hotspots_x = np.append(hotspots_x, np.array(hotspot_result[1]))
                            hotspots_y = np.append(hotspots_y, np.array(hotspot_result[2]))
                            hotspots_z = np.append(hotspots_z, np.array(hotspot_result[3]))
                
    # Once the loop is done we have a list of all hotspots, with duplicates dropped off.

    # Remove the first initialized (0,0)
    hotspots_x = np.delete(hotspots_x, 0)
    hotspots_y = np.delete(hotspots_y, 0)
    hotspots_z = np.delete(hotspots_z, 0)

    # List of hotspots as the original stacked list
    hotspots = np.stack((hotspots_x, hotspots_y, hotspots_z))

    if verbose > 9:
        print("Over:", over, "\nUnder:", under)
        print("Number of seed points:", count)

    # List of hotspots as tuple_list
    hotspots_3d = []
    if as_tuple > 0:
        for i in range(len(hotspots_x)):
            temp_hotspot = (hotspots_x[i], hotspots_y[i], hotspots_z[i])
            hotspots_3d.append(temp_hotspot)
        
        if as_tuple == 1:
            return hotspots_3d
        else:
            return hotspots, hotspots_3d
    else:
        return hotspots
    
    

# image_data matrix (that is 3D) where we want to mark hotspots
# hotspots_3d is a list of hotspots. It is a matrix 3*amount_of_hotspots. Each row is one coordinate.
# area is how large we want to mark the hotspot
# output_directory, where we want to save the new image with marked hotspots.
def markHotspots_3d(image_data_3d_input, hotspots_3d, area, output_directory, verbose=0):

    printed_hotspots = 0

    # So we don't accidentally modify the original image
    image_data_3d = np.copy(image_data_3d_input)

    maxx = np.max(image_data_3d)

    for l in range(np.size(hotspots_3d[0])):

        # Drop hotspots too near the edge so no out of bounds error. Fix hardcode later.
        # if(hotspots_3d[0][l] > image_data_3d.shape[0]-area or hotspots_3d[1][l] > image_data_3d.shape[1]-area or hotspots_3d[2][l] > image_data_3d.shape[2]-area):
        #     image_data_3d[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]] = maxx*1.3

        # Drop all hotspots under certain value
        # Gets rid off all the unwanted background noise hotspots
        # 15000000 for nonscaled
        # ~0.049195 for scaled
        if(image_data_3d[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]] > 0):
            printed_hotspots = printed_hotspots + 1

            for i in range(area):
                for j in range(area):
                    for k in range(area):
                        # If we go over the edge, continue
                        if hotspots_3d[0][l] + i > image_data_3d.shape[0]-1:
                            continue
                        if hotspots_3d[1][l] + j > image_data_3d.shape[1]-1:
                            continue
                        if hotspots_3d[2][l] + k > image_data_3d.shape[2]-1:
                            continue

                        image_data_3d[hotspots_3d[0][l] + i][hotspots_3d[1][l] + j][hotspots_3d[2][l] + k] = maxx*1.3

    if verbose == 1:
        print("Printed hotspots:", printed_hotspots)
        #print( image_data_3d_scaled[77][48][28])

    nifti = nibabel.Nifti1Image(image_data_3d, np.eye(4))
    nibabel.nifti1.save(nifti, output_directory)


def fillHotspots_3d(image_data_3d, hotspots_3d, output_directory, filter=0.0):
    '''
    Color all hotspots that pass the filter ( hotspot_value > filter*maxx).
    '''


    filled_hotspots = 0

    maxx = np.max(image_data_3d)

    # Go through all the hotspots
    for l in range(np.size(hotspots_3d[0])):

        # Drop all hotspots under certain value (0.1*maxx)
        if(image_data_3d[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]] > filter*maxx):

            filled_hotspots = filled_hotspots + 1

            hotspot_value = image_data_3d[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]]

            tol = 0

            # TODO: Create a plot of how steep the hotspot color curve is
            #
            # TODO: Parameter for tolerance is not very good yet. Need to find better way to generate it.
            # Create a steep function for tolerance

            # working for the first three images
            # if ((hotspot_value/maxx) > 0.33):
            #     tol = (hotspot_value)*0.7
            # if ((hotspot_value/maxx) <= 0.33 and (hotspot_value/maxx) >= 0.2):
            #     tol = hotspot_value*0.3
            # if ((hotspot_value/maxx) < 0.2):
            #     tol = hotspot_value*0.15


            if ((hotspot_value/maxx) > 0.7):
                tol = (hotspot_value)*0.6
                # tol = (hotspot_value)*0.15
            if ((hotspot_value/maxx) <= 0.7 and (hotspot_value/maxx) >= 0.4):
                # tol = hotspot_value*0.25
                tol = hotspot_value*0.135
            if ((hotspot_value/maxx) < 0.4):
                tol = hotspot_value*0.08
            

            # Flood fill here
            # print("Original hotspot value:", image_data_3d[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]], "   Tolerance:", tol)
            flood_fill(image_data_3d, (hotspots_3d[0][l], hotspots_3d[1][l], hotspots_3d[2][l]), maxx, tolerance=tol, in_place = True)

    print("Filled hotspots:", filled_hotspots)
    #print( image_data_3d_scaled[77][48][28])

    # Save the final image
    nifti = nibabel.Nifti1Image(image_data_3d, np.eye(4))
    nibabel.nifti1.save(nifti, output_directory)


def getColoredImage(image_data_4d, output_directory, clear_tol=0.025, rat_color=0.0, threshold=0.0, top_brightest=10, factor=0.5, constant=0.05):
    '''### Returns and saves an image that has each hotspot area colored a different value.
    
    image_data_4d       = original image data from dicom file

    output_directory    = where to save the new image

    clear_tol           = tolerance used to clear the background of image.
    Tolerance is in percents of maximum value in the image. Default value 0.025 means that any values within 2.5% tolerance in both directions, close to value in (70,0,70) are set to 0.

    rat_color           = intensity of the nonhotspot areas of rat. Set to 0 to show only hotspots.

    threshold           = search hotspots above this value. Used to drop useless background hotspots

    top_brightest       = choose top n brightest hotspots (used to drop useless background hotspots)

    factor              = a in a*x + b

    constant            = b in a*x + b'''

    # Create two separate images here:
    # 1. base that is used to find hotspot areas 
    # 2. image that contains the hotspot areas colored with different values, the final product

    # Colored image
    image_data_3d_base = np.sum(np.copy(image_data_4d), axis = 3)
    image_data_3d_color = modifyImageData_3D(image_data_3d_base, clear_background=True, tol=clear_tol, verbose=0)#np.zeros((image_data_3d_base.shape[0], image_data_3d_base.shape[1], image_data_3d_base.shape[2]))
    
    # Base image
    image_data_3d_base = np.sum(np.copy(image_data_4d), axis = 3)
    image_data_3d_base = modifyImageData_3D(image_data_3d_base, clear_background=True, tol=clear_tol, verbose=0)

    # Get the hotspots list
    hotspots_3d = findHotspots_3d(image_data_3d_base, 8, threshold)
    print("Hotspots found", hotspots_3d.shape[1])


    # The intensity of nonhotspot areas. Set to 0 to show only hotspots.
    image_data_3d_color = image_data_3d_color * rat_color

    maxx = np.max(image_data_3d_base)

    # print("Hotspots:", hotspots_3d.shape[1])

    printed_hotspots=0

    hotspot_sizes = []
    hotspot_order = []
    hotspot_brightness = []

    chosen_hotspots = 0

    # First loop to determine hotspot sizes for coloring. Can be skipped for simpler colorscheme.
    # ie. bigger hotspot -> lower color value

    rangee = hotspots_3d[0].shape[0]    # 'rangee' because 'range' is a keyword in Python
    l = 0
    while l < rangee:

        # Copy the base image, we need the original image so that the already filled hotspots dont interfere.
        image_data_3d_temp = np.copy(image_data_3d_base)

        # Filter hotspots under 35% color value. Useless if?
        if(image_data_3d_base[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]] > threshold*maxx):
            
            printed_hotspots = printed_hotspots + 1

            hotspot_value = image_data_3d_base[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]]

            tol = 0

            percentage = (factor*(hotspot_value/maxx)**1) + constant
            tol = percentage * hotspot_value
            # print(percentage)

            # Flood fill here
            flood_fill(image_data_3d_temp, (hotspots_3d[0][l], hotspots_3d[1][l], hotspots_3d[2][l]), 2*maxx, tolerance=tol, in_place = True)

            size = len(np.where(image_data_3d_temp==2*maxx)[0])

            # Drop too small hotspot areas.
            # if (size < 20 and hotspots_3d[2][l] == 0) or size < 20:
            if size < 1:
                # print(f"Zero at {l}", hotspots_3d[2][l])

                # Delete the rejected hotspot from the hotspots nplist
                hotspots_3d = np.delete(hotspots_3d, l, axis=1)
                # So we dont mess up the loop
                rangee = rangee-1
                continue
            
            chosen_hotspots = chosen_hotspots + 1
            # hotspot_sizes.append(size)
            hotspot_brightness.append(image_data_3d_base[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]])
            
        l = l + 1

    # Simple hotspot filter
    # Choose only the 10 brightest hotspots

    hotspot_order = np.arange(0, chosen_hotspots)

    # List of all hotspot values
    hotspot_brightness = np.array(hotspot_brightness)

    # Sort the hotspots by brightness (how high value they contain).
    p = np.argsort(hotspot_brightness)
    new_hotspot_order = hotspot_order[p]

    # For example: choose only the 10 brightest hotspots
    hotspot_amount = top_brightest
    if(new_hotspot_order.shape[0] > hotspot_amount):
        new_hotspot_order = new_hotspot_order[new_hotspot_order.shape[0]-hotspot_amount : ]


    colored_hotspots = 0

    # Coloring loop. Iterate through all hotspots that passed the filter.
    for hs in new_hotspot_order:

        # Copy fresh base image so that older coloring doesnt affect
        image_data_3d_temp = np.copy(image_data_3d_base)

        # Filter hotspots under 35% color value. Useless if? Above filter makes obsolete. Kept if the implementation changes.
        # if(image_data_3d_base[hotspots_3d[0][hs]][hotspots_3d[1][hs]][hotspots_3d[2][hs]] > threshold*maxx):

        hotspot_value = image_data_3d_base[hotspots_3d[0][hs]][hotspots_3d[1][hs]][hotspots_3d[2][hs]]
        tol = 0

        percentage = (factor*(hotspot_value/maxx)**1) + constant
        tol = percentage * hotspot_value
        # print(percentage)

        # Limit the area where coloring happens.
        # We take a certain area around the hotspot and take a copy of the image and apply coloring there. Then move this coloring to the final image.
        # Not implemented yet

        # # What is this :DDD
        # zeros = np.where(True, 0, image_data_3d_temp)
        # # print(zeros)
        # # image_data_3d_temp = image_data_3d_temp[hotspots_3d[0][hs] - 10 : hotspots_3d[0][hs] + 10, hotspots_3d[1][hs] - 10 : hotspots_3d[1][hs] + 10, hotspots_3d[2][hs] - 10 : hotspots_3d[2][hs] + 10]

        # color_area = 3

        # zeros[hotspots_3d[0][hs] - color_area : hotspots_3d[0][hs] + color_area, hotspots_3d[1][hs] - color_area : hotspots_3d[1][hs] + color_area, hotspots_3d[2][hs] - color_area : hotspots_3d[2][hs] + color_area] = image_data_3d_temp[hotspots_3d[0][hs] - color_area : hotspots_3d[0][hs] + color_area, hotspots_3d[1][hs] - color_area : hotspots_3d[1][hs] + color_area, hotspots_3d[2][hs] - color_area : hotspots_3d[2][hs] + color_area]

        # print(np.where(zeros > 0)[0].shape)

        # Flood fill this particular hotspot here. As it is done to the fresh base image there are no other hotspots interfering.
        flood_fill(image_data_3d_temp, (hotspots_3d[0][hs], hotspots_3d[1][hs], hotspots_3d[2][hs]), 2*maxx, tolerance=tol, in_place = True)

        # Set color (value) here
        color = maxx*colored_hotspots*(1/len(new_hotspot_order))#*(1-rat_color) + maxx*rat_color
        # print(color)

        # Coloring happens here. Check where in the temp image values are 2*maxx (impossible to occur naturally) and color corresponding area with current color.
        image_data_3d_color = np.where(image_data_3d_temp == 2*maxx, color, image_data_3d_color)

        colored_hotspots = colored_hotspots + 1

    print("Colored hotspots:", colored_hotspots)

    # Save the final image
    nifti = nibabel.Nifti1Image(image_data_3d_color, np.eye(4))
    nibabel.nifti1.save(nifti, output_directory)

    return image_data_3d_color



def getSeparateOrgans(image_data_4d, output_directory, clear_tol=0.025, threshold=0.0, top_brightest=10, factor=0.5, constant=0.05):
    '''### Returns each hotspot area in a separate file.
    Outputs a binary mask of each hotspotarea.
    
    image_data_4d       = original image data from dicom file

    output_directory    = where to save the new image

    clear_tol           = tolerance used to clear the background of image.
    Tolerance is in percents of maximum value in the image. Default value 0.025 means that any values within 2.5% tolerance in both directions, close to value in (70,0,70) are set to 0.

    rat_color           = intensity of the nonhotspot areas of rat. Set to 0 to show only hotspots.

    threshold           = search hotspots above this value. Used to drop useless background hotspots

    top_brightest       = choose top n brightest hotspots (used to drop useless background hotspots)

    factor              = a in a*x^2 + b

    constant            = b in a*x^2 + b'''

    # Create two separate images here:
    # 1. base that is used to find hotspot areas 
    # 2. image that contains the hotspot areas colored with different values, the final product

    # Colored image
    image_data_3d_base = np.sum(np.copy(image_data_4d), axis = 3)
    
    # Base image
    image_data_3d_base = np.sum(np.copy(image_data_4d), axis = 3)
    image_data_3d_base = modifyImageData_3D(image_data_3d_base, clear_background=True, tol=clear_tol, verbose=0)

    # Get the hotspots list
    hotspots_3d = findHotspots_3d(image_data_3d_base, 8, threshold)
    print("Hotspots found", hotspots_3d.shape[1])

    maxx = np.max(image_data_3d_base)

    # print("Hotspots:", hotspots_3d.shape[1])

    printed_hotspots=0

    hotspot_sizes = []
    hotspot_order = []
    hotspot_brightness = []

    chosen_hotspots = 0

    # First loop to determine hotspot sizes for coloring. Can be skipped for simpler colorscheme.
    # ie. bigger hotspot -> lower color value

    rangee = hotspots_3d[0].shape[0]    # 'rangee' because 'range' is a keyword in Python
    l = 0
    while l < rangee:

        # Copy the base image, we need the original image so that the already filled hotspots dont interfere.
        image_data_3d_temp = np.copy(image_data_3d_base)

        # Filter hotspots under 35% color value. Useless if?
        if(image_data_3d_base[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]] > threshold*maxx):
            
            printed_hotspots = printed_hotspots + 1

            hotspot_value = image_data_3d_base[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]]

            tol = 0

            percentage = (factor*(hotspot_value/maxx)**1) + constant
            tol = percentage * hotspot_value
            # print(percentage)

            # Flood fill here
            flood_fill(image_data_3d_temp, (hotspots_3d[0][l], hotspots_3d[1][l], hotspots_3d[2][l]), 2*maxx, tolerance=tol, in_place = True)

            size = len(np.where(image_data_3d_temp==2*maxx)[0])

            # Drop too small hotspot areas.
            # if (size < 20 and hotspots_3d[2][l] == 0) or size < 20:
            if size < 1:
                # print(f"Zero at {l}", hotspots_3d[2][l])

                # Delete the rejected hotspot from the hotspots nplist
                hotspots_3d = np.delete(hotspots_3d, l, axis=1)
                # So we dont mess up the loop
                rangee = rangee-1
                continue
            
            chosen_hotspots = chosen_hotspots + 1
            # hotspot_sizes.append(size)
            hotspot_brightness.append(image_data_3d_base[hotspots_3d[0][l]][hotspots_3d[1][l]][hotspots_3d[2][l]])
            
        l = l + 1

    # Simple hotspot filter
    # Choose only the 10 brightest hotspots

    hotspot_order = np.arange(0, chosen_hotspots)

    # List of all hotspot values
    hotspot_brightness = np.array(hotspot_brightness)

    # Sort the hotspots by brightness (how high value they contain).
    p = np.argsort(hotspot_brightness)
    new_hotspot_order = hotspot_order[p]

    # For example: choose only the 10 brightest hotspots
    hotspot_amount = top_brightest
    if(new_hotspot_order.shape[0] > hotspot_amount):
        new_hotspot_order = new_hotspot_order[new_hotspot_order.shape[0]-hotspot_amount : ]


    colored_hotspots = 0

    # Coloring loop. Iterate through all hotspots that passed the filter.
    for hs in new_hotspot_order:

        # Copy fresh base image so that older coloring doesnt affect
        image_data_3d_temp = np.copy(image_data_3d_base)
        image_data_3d_color = np.zeros((image_data_3d_base.shape[0], image_data_3d_base.shape[1], image_data_3d_base.shape[2]))

        # Filter hotspots under 35% color value. Useless if? Above filter makes obsolete. Kept if the implementation changes.
        # if(image_data_3d_base[hotspots_3d[0][hs]][hotspots_3d[1][hs]][hotspots_3d[2][hs]] > threshold*maxx):

        hotspot_value = image_data_3d_base[hotspots_3d[0][hs]][hotspots_3d[1][hs]][hotspots_3d[2][hs]]
        tol = 0

        percentage = (factor*(hotspot_value/maxx)**1) + constant
        tol = percentage * hotspot_value
        # print(percentage)

        # Limit the area where coloring happens.
        # We take a certain area around the hotspot and take a copy of the image and apply coloring there. Then move this coloring to the final image.
        # Not implemented yet

        # # What is this :DDD
        # zeros = np.where(True, 0, image_data_3d_temp)
        # # print(zeros)
        # # image_data_3d_temp = image_data_3d_temp[hotspots_3d[0][hs] - 10 : hotspots_3d[0][hs] + 10, hotspots_3d[1][hs] - 10 : hotspots_3d[1][hs] + 10, hotspots_3d[2][hs] - 10 : hotspots_3d[2][hs] + 10]

        # color_area = 3

        # zeros[hotspots_3d[0][hs] - color_area : hotspots_3d[0][hs] + color_area, hotspots_3d[1][hs] - color_area : hotspots_3d[1][hs] + color_area, hotspots_3d[2][hs] - color_area : hotspots_3d[2][hs] + color_area] = image_data_3d_temp[hotspots_3d[0][hs] - color_area : hotspots_3d[0][hs] + color_area, hotspots_3d[1][hs] - color_area : hotspots_3d[1][hs] + color_area, hotspots_3d[2][hs] - color_area : hotspots_3d[2][hs] + color_area]

        # print(np.where(zeros > 0)[0].shape)

        # Flood fill this particular hotspot here. As it is done to the fresh base image there are no other hotspots interfering.
        flood_fill(image_data_3d_temp, (hotspots_3d[0][hs], hotspots_3d[1][hs], hotspots_3d[2][hs]), 2*maxx, tolerance=tol, in_place = True)

        # Coloring happens here. Check where in the temp image values are 2*maxx (impossible to occur naturally) and color corresponding area with 1.
        image_data_3d_color = np.where(image_data_3d_temp == 2*maxx, 1, image_data_3d_color)

        # hotspot_str = str((hotspots_3d[0][hs], hotspots_3d[1][hs], hotspots_3d[2][hs]))
        hotspot_str = str(hotspots_3d[0][hs]) + "_" + str(hotspots_3d[1][hs]) + "_" + str(hotspots_3d[2][hs])

        # Save the final image
        nifti = nibabel.Nifti1Image(image_data_3d_color, np.eye(4))
        nibabel.nifti1.save(nifti, (output_directory + "_" + hotspot_str))

        colored_hotspots = colored_hotspots + 1

    print("Saved hotspots:", colored_hotspots)




def FixPics(input_path, output_path, id, organ, verbose=0):

    final_input_path = input_path + "/" + id + "_" + organ + ".img"
    if verbose > 0:
        print("Nonfixed image directory: ", final_input_path)

    try:
        img = nibabel.load(final_input_path)
    except:
        print("Path is invalid.")
        return

    # Load data
    img_data = img.get_fdata()

    # Swap x- and y-axes
    swapped_img = np.swapaxes(img_data, 0, 1)
    # Invert z-axis
    fixed_img = np.flip(swapped_img, 2)

    final_output_path = output_path + "/fixed_" + organ
    if verbose > 0:
        print("Fixed image directory: ", final_output_path)

    if type(output_path) != int:
        nifti = nibabel.Nifti1Image(fixed_img, np.eye(4))
        nibabel.nifti1.save(nifti, final_output_path)

    return fixed_img

def FixPics2(input_path, output_path, verbose=0):

    if verbose > 0:
        print("Nonfixed image directory: ", input_path)

    try:
        img = nibabel.load(input_path)
    except:
        print("Path is invalid.")
        return

    # Load data
    img_data = img.get_fdata()

    # Swap x- and y-axes
    swapped_img = np.swapaxes(img_data, 0, 1)
    # Invert z-axis
    fixed_img = np.flip(swapped_img, 2)

    if verbose > 0:
        print("Fixed image directory: ", output_path)

    if type(output_path) != int:
        nifti = nibabel.Nifti1Image(fixed_img, np.eye(4))
        nibabel.nifti1.save(nifti, output_path)

    return fixed_img

def JaccardIndex(img1, img2, verbose=0):
    '''Check each index if they are both > 0
    '''
    m00 = 0 # A = 1, B = 0
    m01 = 0 # A = 0, B = 1
    m10 = 0 # A = 1, B = 0
    m11 = 0 # A = 1, B = 1

    for i in range (img1.shape[0]):
        for j in range (img1.shape[1]):
            for k in range (img1.shape[2]):

                # if img1[i][j][k] == 0 and  img2[i][j][k] == 0 :
                #     m00 = m00 + 1

                if img1[i][j][k] == 0 and img2[i][j][k] > 0 :
                    m01 = m01 + 1

                if img1[i][j][k] > 0 and img2[i][j][k] == 0 :
                    m10 = m10 + 1

                if img1[i][j][k] > 0 and img2[i][j][k] > 0 :
                    m11 = m11 + 1

    if (m01 + m10 + m11) == 0:
        return 10

    

    jaccard_similarity_index = m11 / (m01 + m10 + m11)
    if verbose > 0:
        print(jaccard_similarity_index)

    return jaccard_similarity_index

def FindOrgan(ref_organ, hotspot_list, use_bbox=0):
    """Finds hotspots that are within a bounding box of the reference organ

    hotspot_list as (3, n) sized list (0, n)=x, (1,n)=y, (2,n)=z

    reference_organ = path to the reference image
    """
    if use_bbox > 0:
        # Load ref image data
        ref_organ_image_data = nibabel.load(ref_organ).get_fdata()


        # Get all positive voxels coordinates
        all_pos_voxels = np.where(ref_organ_image_data > 0)

        # print(all_pos_voxels[0].shape)

        if all_pos_voxels[0].shape[0] == 0:
            return []

        # Get bounding box max values
        x_min = np.min(all_pos_voxels[0])
        x_max = np.max(all_pos_voxels[0])

        y_min = np.min(all_pos_voxels[1])
        y_max = np.max(all_pos_voxels[1])

        z_min = np.min(all_pos_voxels[2])
        z_max = np.max(all_pos_voxels[2])

        # print(x_min, x_max, y_min, y_max, z_min, z_max)

        # Create new zeros array where we save our new bounding box
        bbox = np.zeros([ref_organ_image_data.shape[0], ref_organ_image_data.shape[1], ref_organ_image_data.shape[2]])
        # print(bbox.shape)

        # Mask the bounding box to 1
        bbox[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1

        # The bounding box coordinates that have the value = 1
        bbox_coords = np.where(bbox == 1)

        # Save the bounding box (debug)
        # nifti = nibabel.Nifti1Image(bbox, np.eye(4))
        # nibabel.nifti1.save(nifti, "bounding_box_test")

        new_hotspot_list = []

        # Check which hotspots are inside the bounding box
        # Iterate through all hotspots
        for i in range(len(hotspot_list[0])):

            # Iterate through all bounding box coordinates
            for j in range(bbox_coords[0].shape[0]):
                # Check if all the coordinates of hotspot_list match the one in the bounding box
                if hotspot_list[0][i] == bbox_coords[0][j] and hotspot_list[1][i] == bbox_coords[1][j] and hotspot_list[2][i] == bbox_coords[2][j]:
                    # print(hotspot_list[0][i], hotspot_list[1][i], hotspot_list[2][i])
                    new_hotspot_list.append((hotspot_list[0][i], hotspot_list[1][i], hotspot_list[2][i]))

        # print(np.where(ref_organ_image_data == True)[0])
        return new_hotspot_list

    # Don't use bounding box, just the original volumes
    else:
        # Load ref image data
        ref_organ_image_data = nibabel.load(ref_organ).get_fdata()


        # Get all positive voxels coordinates
        all_pos_voxels = np.where(ref_organ_image_data > 0)

        new_hotspot_list = []

        print(all_pos_voxels[0].shape)

        # Check which hotspots are inside the bounding box
        # Iterate through all hotspots
        for i in range(len(hotspot_list[0])):

            # Iterate through all bounding box coordinates
            for j in range(all_pos_voxels[0].shape[0]):
                # Check if all the coordinates of hotspot_list match the one in the bounding box
                if hotspot_list[0][i] == all_pos_voxels[0][j] and hotspot_list[1][i] == all_pos_voxels[1][j] and hotspot_list[2][i] == all_pos_voxels[2][j]:
                    # print(hotspot_list[0][i], hotspot_list[1][i], hotspot_list[2][i])
                    new_hotspot_list.append((hotspot_list[0][i], hotspot_list[1][i], hotspot_list[2][i]))

        return new_hotspot_list

def ColorSelectedHotspots(image_data_4d, hotspot_list, output_directory, clear_tol=0.025, threshold=0.0, factor=0.5, constant=0.05, direct_tolerance=0):
    '''### Returns an image that masks certain hotspots to 1
    Outputs a binary mask of each hotspotarea in one picture.
    
    image_data_4d       = original image data from dicom file

    hotspot_list        = list of tuples (x, y, z)

    output_directory    = where to save the new image

    clear_tol           = tolerance used to clear the background of image.
    Tolerance is in percents of maximum value in the image. Default value 0.025 means that any values within 2.5% tolerance in both directions, close to value in (70,0,70) are set to 0.

    threshold           = search hotspots above this value. Used to drop useless background hotspots

    factor              = a in a*x + b

    constant            = b in a*x + b'''

    # Create two separate images here:
    # 1. base that is used to find hotspot areas 
    # 2. image that contains the hotspot areas colored with different values, the final product

    # Colored image
    image_data_3d_base = np.sum(np.copy(image_data_4d), axis = 3)
    
    # Base image
    image_data_3d_base = np.sum(np.copy(image_data_4d), axis = 3)
    image_data_3d_base = modifyImageData_3D(image_data_3d_base, clear_background=True, tol=clear_tol, verbose=0)

    maxx = np.max(image_data_3d_base)

    colored_hotspots = 0

    # Final colored image
    image_data_3d_color = np.zeros((image_data_3d_base.shape[0], image_data_3d_base.shape[1], image_data_3d_base.shape[2]))

    # Coloring loop. Iterate through all hotspots that passed the filter.
    for hs in range(len(hotspot_list)):

        # print(hotspot_list[hs][0], hotspot_list[hs][1], hotspot_list[hs][2])

        # Copy fresh base image so that older coloring doesnt affect
        image_data_3d_temp = np.copy(image_data_3d_base)

        hotspot_value = image_data_3d_base[hotspot_list[hs][0]][hotspot_list[hs][1]][hotspot_list[hs][2]]
        tol = 0

        percentage = (factor*(hotspot_value/maxx)**1) + constant
        tol = percentage * hotspot_value
        if direct_tolerance != 0:
            tol = hotspot_value * direct_tolerance
        # print(percentage)

        # Flood fill this particular hotspot here. As it is done to the fresh base image there are no other hotspots interfering.
        flood_fill(image_data_3d_temp, (hotspot_list[hs][0], hotspot_list[hs][1], hotspot_list[hs][2]), 2*maxx, tolerance=tol, in_place = True, connectivity=None)

        # Coloring happens here. Check where in the temp image values are 2*maxx (impossible to occur naturally) and color corresponding area with 1.
        image_data_3d_color = np.where(image_data_3d_temp == 2*maxx, 1, image_data_3d_color)

        colored_hotspots = colored_hotspots + 1

    # Save the final image
    nifti = nibabel.Nifti1Image(image_data_3d_color, np.eye(4))
    nibabel.nifti1.save(nifti, (output_directory))

    # print("Saved hotspots:", colored_hotspots)

    return image_data_3d_color

def SeparateKidneys(image_path, output_path, verbose=0):
    '''Separates kidneys in to two different images from the manual segmentation where they are together.
    '''

    if verbose > 0:
        print('Image path: ', os.getcwd(), sep='')
        print('Output folder:', output_path, sep='')

    fixed_kidneys = FixPics2(image_path, output_path+"/fixed_kidney.nii", verbose=verbose)

    # Check where the mask starts and ends
    start_index = np.where(fixed_kidneys > 0)

    # Check where there is a complete gap, split there to 2 pictures
    for i in range(np.min(start_index[1]), np.max(start_index[1])+5):
        temp_slice = fixed_kidneys[:, i, :]
        positive_indices = np.where(temp_slice > 0)
        if positive_indices[0].shape[0] == 0:
            
            # Create the new kidney image
            right_kidney = fixed_kidneys.copy()
            right_kidney[:, i+1:, :] = 0
            nifti = nibabel.Nifti1Image(right_kidney, np.eye(4))
            nibabel.nifti1.save(nifti, output_path + "/kidney_right.nii")

            # Create the new kidney image
            left_kidney = fixed_kidneys.copy()
            left_kidney[:, :i+1, :] = 0
            nifti = nibabel.Nifti1Image(left_kidney, np.eye(4))
            nibabel.nifti1.save(nifti, output_path + "/kidney_left.nii")

            return right_kidney, left_kidney


    