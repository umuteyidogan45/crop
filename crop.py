import cv2
import numpy as np
import os
import random

def choose_random_side():
    return random.choice(['left', 'right'])

def find_intersection(line1, line2):
    A = np.array([[line1[0], -1], [line2[0], -1]])
    B = np.array([-line1[1], -line2[1]])
    intersection = np.linalg.solve(A, B)
    return intersection.astype(int)

def get_slope_intercept(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        slope = "vertical"
        intercept = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    return slope, intercept

def intersec(now, next, l1, l2):
    s1, i1 = get_slope_intercept(now, next)
    s2, i2 = get_slope_intercept(l1, l2)
    if s1 == "vertical":
        x = i1
        y = s2 * x + i2
        return np.array([x,y]).astype(int)
    if s2 == "vertical":
        x = i2
        y = s1 * x + i1
        return np.array([x,y]).astype(int)
    return find_intersection((s1,i1), (s2,i2))


def parse_label_line(line, image_width, image_height):
    parts = line.strip().split()
    label_class = int(parts[0])
    normalized_coordinates = [float(coord) for coord in parts[1:]]
    
    pixel_coordinates = []
    for i in range(0, len(normalized_coordinates), 2):
        x = normalized_coordinates[i] 
        y = normalized_coordinates[i + 1] 
        pixel_coordinates.append((x, y))
    
    polygon = np.array(pixel_coordinates, dtype=np.int32)
    return label_class, pixel_coordinates

def read_label_file(file_path, image_width, image_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = [parse_label_line(line, image_width, image_height) for line in lines]
    return labels

def crop_image_and_labels(image, labels, xmin, xmax, ymin, ymax):
    cropped_image = image[ymin:ymax, xmin:xmax]
    adjusted_labels = []
    up = [(0, ymax), (10, ymax)]
    down = [(0, ymin), (10, ymin)]
    left = [(xmin, 0), (xmin, 10)]
    right = [(xmax, 0), (xmax, 10)]

    for label_class, points in labels:
        adjusted_points = []
        adjusted_points2 = []
        adjusted_points3 = []
        adjusted_points4 = []
        inout = []
        for a in range(len(points)):
            if xmax >= points[a][0]:
                inout.append(1) 
            else:
                inout.append(0) 

        r1 = np.array([0,0])
        r2 = np.array([0,0])

        for a in range(len(points)):
            if xmax >= points[a][0]:
                adjusted_points.append((points[a][0], points[a][1]))
            else:
                if (inout[a-1]):
                    r1 = intersec(points[a], points[a-1], right[0], right[1])
                    if (xmax < points[a][0]):
                        adjusted_points.append((r1[0], r1[1]))
                if (a == len(points) - 1):
                    if (inout[0]):
                        r2 = intersec(points[a], points[0], right[0], right[1])
                        if (xmax < points[a][0]):
                            adjusted_points.append((r2[0], r2[1]))
                elif (inout[a+1]):
                    r2 = intersec(points[a], points[a+1], right[0], right[1])
                    if (xmax < points[a][0]):
                        adjusted_points.append((r2[0], r2[1]))
        
        inout.clear()

        for a in range(len(adjusted_points)):
            if xmin <= adjusted_points[a][0]:
                inout.append(1) #in
            else:
                inout.append(0) #out

        for a in range(len(adjusted_points)):
            if xmin <= adjusted_points[a][0]:
                adjusted_points2.append((adjusted_points[a][0], adjusted_points[a][1]))
            else:
                if (inout[a-1]):
                    r1 = intersec(adjusted_points[a], adjusted_points[a-1], left[0], left[1])
                    if (xmin > adjusted_points[a][0]):
                        adjusted_points2.append((r1[0], r1[1]))
                if (a == len(adjusted_points) - 1):
                    if (inout[0]):
                        r2 = intersec(adjusted_points[a], adjusted_points[0], left[0], left[1])
                        if (xmin >  adjusted_points[a][0]):
                            adjusted_points2.append((r2[0], r2[1]))
                elif (inout[a+1]):
                    r2 = intersec(adjusted_points[a], adjusted_points[a+1], left[0], left[1])
                    if (xmin > adjusted_points[a][0]):
                        adjusted_points2.append((r2[0], r2[1]))
            
        inout.clear()

        for a in range(len(adjusted_points2)):
            if adjusted_points2[a][1] <= ymax:
                inout.append(1) #in
            else:
                inout.append(0) #out

        for a in range(len(adjusted_points2)):
            if adjusted_points2[a][1] <= ymax:
                adjusted_points3.append((adjusted_points2[a][0], adjusted_points2[a][1]))
            else:
                if (inout[a-1]):
                    r1 = intersec(adjusted_points2[a], adjusted_points2[a-1], up[0], up[1])
                    if (ymax < adjusted_points2[a][1]):
                        adjusted_points3.append((r1[0], r1[1]))
                if (a == len(adjusted_points2) - 1):
                    if (inout[0]):
                        r2 = intersec(adjusted_points2[a], adjusted_points2[0], up[0], up[1])
                        if (ymax <  adjusted_points2[a][1]):
                            adjusted_points3.append((r2[0], r2[1]))
                elif (inout[a+1]):
                    r2 = intersec(adjusted_points2[a], adjusted_points2[a+1], up[0], up[1])
                    if (ymax < adjusted_points2[a][1]):
                        adjusted_points3.append((r2[0], r2[1]))

        inout.clear()

        for a in range(len(adjusted_points3)):
            if ymin <= adjusted_points3[a][1]:
                inout.append(1) #in
            else:
                inout.append(0) #out

        for a in range(len(adjusted_points3)):
            if ymin <= adjusted_points3[a][1]:
                adjusted_points4.append((adjusted_points3[a][0], adjusted_points3[a][1]))
            else:
                if (inout[a-1]):
                    r1 = intersec(adjusted_points3[a], adjusted_points3[a-1], down[0], down[1])
                    if (ymin > adjusted_points3[a][1]):
                        adjusted_points4.append((r1[0], r1[1]))
                if (a == len(adjusted_points3) - 1):
                    if (inout[0]):
                        r2 = intersec(adjusted_points3[a], adjusted_points3[0], down[0], down[1])
                        if (ymin >  adjusted_points3[a][1]):
                            adjusted_points4.append((r2[0], r2[1]))
                elif (inout[a+1]):
                    r2 = intersec(adjusted_points3[a], adjusted_points3[a+1], down[0], down[1])
                    if (ymin > adjusted_points3[a][1]):
                        adjusted_points4.append((r2[0], r2[1]))

        for a in range(len(adjusted_points4)):
            adjusted_points4[a] = (adjusted_points4[a][0] - xmin, adjusted_points4[a][1] - ymin)

        if adjusted_points4:
            adjusted_labels.append((label_class, adjusted_points4))
    return cropped_image, adjusted_labels

def cordfinder(image, labels):
    image_height, image_width, _ = image.shape
    center = [0,0]
    labelclass, points = labels[0]
    for a in points:
        center[0] += a[0]
        center[1] += a[1]
    center[0] //= len(points)
    center[1] //= len(points)
    xmin = random.randrange(0, center[0])
    xmax = random.randrange(center[0], image_width)
    ymin = random.randrange(0, center[1])
    ymax = random.randrange(center[1], image_height)
    
    return xmin,xmax,ymin,ymax


def convert_coordinates(normalized_coordinates, image_width, image_height):
    final = []
    
    for a, poly in normalized_coordinates:
        pixel_coordinates = []
        for i in range(0, len(poly)):
            x = int(poly[i][0] * image_width)
            y = int(poly[i][1] * image_height)
            pixel_coordinates.append((x, y))
        final.append((a, pixel_coordinates))
    return final


def testcrop(image_path, rnum):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    name = os.path.basename(image_path)
    beee = os.path.splitext(name)[0]
    label_file_path = "/labels/" + beee + ".txt"
    normlabels = read_label_file(label_file_path, image_width, image_height)
    crop(image_path, normlabels, ".", rnum)



def findframe(polygon):
    xmin = 100000
    ymin = 100000
    xmax = 0
    ymax = 0
    for i in range(len(polygon)):
            if xmin > polygon[i][0]:
                xmin = polygon[i][0]
            if xmax < polygon[i][0]:
                xmax = polygon[i][0]
            if ymin > polygon[i][1]:
                ymin = polygon[i][1]
            if ymax < polygon[i][1]:
                ymax = polygon[i][1]
    return (xmin,xmax,ymin,ymax)

def calculate_polygon_area(polygon_points):
    return cv2.contourArea(np.array(polygon_points))


def find_farthest_scratch(labels, side): #retruns both the list of farthest points and the farthest label coordinates according the chosen side
    def find_max_x(coordinates):
        max_x = float('-inf')  # Initialize with negative infinity to ensure any x value is greater
        for x, y in coordinates:
            if x > max_x:
                max_x = x
        return max_x
    
    def find_min_x(coordinates):
        min_x = float('inf')  # Initialize with positive infinity to ensure any x value is smaller
        for x, y in coordinates:
            if x < min_x:
                min_x = x
        return min_x
    
    def find_index_of_max_value(input_list):
        max_value = max(input_list)
        max_index = input_list.index(max_value)
        return max_index
    
    def find_index_of_min_value(input_list):
        min_value = min(input_list)
        min_index = input_list.index(min_value)
        return min_index
    
    def find_max_y(coordinates):
        max_y = float('-inf')  # Initialize with negative infinity to ensure any x value is greater
        for x, y in coordinates:
            if y > max_y:
                max_y = y
        return max_y
    
    def find_min_y(coordinates):
        min_y = float('inf')  # Initialize with positive infinity to ensure any x value is smaller
        for x, y in coordinates:
            if y < min_y:
                min_y = y
        return min_y
    
    if side == 'left':
        labels_max_x = []
        for class_id, label in labels:   
            max_x = find_max_x(label)
            labels_max_x.append(max_x)
        farthest_label_index = find_index_of_max_value(labels_max_x)
        return labels_max_x, labels[farthest_label_index]
    
    elif side == 'right':
        labels_min_x = []
        for class_id, label in labels:   
            min_x = find_min_x(label)
            labels_min_x.append(min_x)
        farthest_label_index = find_index_of_min_value(labels_min_x)
        return labels_min_x, labels[farthest_label_index]
    
    elif side == 'up':
        labels_max_y = []
        for class_id, label in labels:   
            max_y = find_max_y(label)
            labels_max_y.append(max_y)
        farthest_label_index = find_index_of_max_value(labels_max_y)
        return labels_max_y, labels[farthest_label_index]
    
    elif side == "bottom":
        labels_min_y = []
        for class_id, label in labels:   
            min_y = find_min_y(label)
            labels_min_y.append(min_y)
        farthest_label_index = find_index_of_min_value(labels_min_y)
        return labels_min_y, labels[farthest_label_index]


def findthres(image, base, threshold, farthest, frame):
    totalarea = calculate_polygon_area(farthest[1])
    
    shiftedbase = base - 1
    _ , croppedpoly = crop_image_and_labels(image, [farthest], shiftedbase, base, frame[2], frame[3])
    croppedarea = calculate_polygon_area(croppedpoly[0][1])
    while ((totalarea - croppedarea)/totalarea) * 100 > threshold:
        shiftedbase -= 1
        _ , croppedpoly = crop_image_and_labels(image, [farthest], shiftedbase, base, frame[2], frame[3])
        croppedarea = calculate_polygon_area(croppedpoly[0][1])

        
    return shiftedbase


def partition(array, low, high):
 
    pivot = array[high]
 
    i = low - 1

    for j in range(low, high):
        if array[j] <= pivot:
 
            i = i + 1
 
            (array[i], array[j]) = (array[j], array[i])
 
    (array[i + 1], array[high]) = (array[high], array[i + 1])
 
    return i + 1
 
 
def quickSort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
 
        quickSort(array, low, pi - 1)
 
        quickSort(array, pi + 1, high)
    return array

def crop_scratchs(coord_list, side):
    length = len(coord_list)
    desired_ratio = 0.25
    number_of_cropping_scratchs = int(desired_ratio*length)
    sorted_arr = quickSort(coord_list, 0, length-1)
    if side == "left":
        return sorted_arr[number_of_cropping_scratchs-1]
    elif side == "right":
        return sorted_arr[length - number_of_cropping_scratchs]
   

def write_labels(output_label_path, labels):
    with open(output_label_path, 'w') as label_file:
        for label in labels:
            label_line = ' '.join(map(str, label)) + '\n'
            label_file.write(label_line)


def crop(image_path, normlabels, output_folder, repeatnum):
    alllabels = []
    for i in range(repeatnum):
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        output_folder_photos = "./croppedphotos"
        output_folder_labels = "./croppedlabels"

        labels = convert_coordinates(normlabels, image_width, image_height)
        side = choose_random_side()

        
        x_coords_list, farthest = find_farthest_scratch(labels, side)
        area_of_scratch = calculate_polygon_area(farthest[1])
        print(area_of_scratch)
        
        scratchframe = findframe(farthest[1])
        print(scratchframe)

        base = findthres(image, scratchframe[1], 70, farthest, scratchframe)

        
        _, lowestscratch = find_farthest_scratch(labels, "up")
        scratchframelow = findframe(lowestscratch[1])

        _, highestscratch = find_farthest_scratch(labels, "bottom")
        scratchframehigh = findframe(highestscratch[1])

        if side == "left":
            xmin = crop_scratchs(x_coords_list, side)
            xmax = random.randint(base, image_width)
        else: 
            xmin = random.randint(0, base)
            xmax = crop_scratchs(x_coords_list, side)

        ymin = random.randint(0, scratchframelow[2])
        ymax = random.randint(0, scratchframehigh[2])
        newwidth = xmax - xmin
        newheight = ymax - ymin

        name = os.path.basename(image_path)
        #print(name)


        cropped_image, adjusted_labels = crop_image_and_labels(image, labels, xmin, xmax, ymin, ymax)

        alllabels.append(adjusted_labels)

        output_image_path = os.path.join(output_folder_photos, name)

        #print("path is: ", output_image_path)
        #print("img is:", cropped_image.shape)

        cv2.imwrite(output_image_path, cropped_image)
        
        output_label_path = os.path.join(output_folder_labels, name + '.txt')
        write_labels(output_label_path, alllabels)

        with open(output_label_path, 'w') as label_output:
            for label_class, points in adjusted_labels:
                label_output.write(f"{label_class} " + " ".join([f"{x/newwidth} {y/newheight}" for x, y in points]) + "\n")
    return alllabels, repeatnum

testcrop("/home/gizem/Desktop/servislet/task_7/crop/images/1162658018_front.jpeg", 1)