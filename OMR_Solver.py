from commonfunctions import *

#block size to look at, offset to be added to the block mean value
def local_binarize(img, block_size = 35, offset_val = 10):
    img=img_as_ubyte(img)
    return img < threshold_otsu(img)
#######################################
def get_max_freq(list):
    freq = {} 
    for item in list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    wanted_value=0
    wanted_angle=0
    for key, value in freq.items():
        if wanted_value<value:
            wanted_value=value
            wanted_angle=key       
    return wanted_angle

def deskew(bw_image):
    # hough line to detect lines in the photo
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(bw_image, theta=tested_angles)
    origin = np.array((0, bw_image.shape[1]))
    # hough peaks to get those lines
    angles=[]
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        angles.append(angle)
    angles = [angle * 180 / np.pi for angle in angles]
    # get the most repeated angle on photo
    rotating_angle=get_max_freq(angles)
    # rotate photo
    
    if rotating_angle == 90:
        return bw_image, rotating_angle
        
    if rotating_angle>0:
        rotated_image=rotate(bw_image,rotating_angle-90,resize=True)
    else:
        rotated_image=rotate(bw_image,rotating_angle+90,resize=True)
    return rotated_image, rotating_angle
#######################################
def find_ref_lengths(img):
    white_runs = []
    black_runs = []
    for col in img.T:
        #add 0 at the start and end to handle the length
        bounded = np.hstack(([0], col, [0]))
        #get the first difference
        difs = np.diff(bounded)
        #a run of ones starts at (1) and ends at (-1)
        run_starts, = np.where(difs > 0)
        run_ends, = np.where(difs < 0)
        
        #Append the runs:
        white_runs.extend(run_ends - run_starts)
        black_runs.extend(run_starts[1:] - run_ends[:-1])
    
    line_width = np.bincount(white_runs).argmax()
    line_spacing = np.bincount(black_runs).argmax()
     
    return line_width, line_spacing, white_runs
######################################
def remove_symbols(image, thresh):
    line_width, line_spacing, white_runs = find_ref_lengths(image)
    m=0
    j=0
    img= np.zeros(image.shape)
    for i in range(image.shape[1]):
        j=0
        while j<(image.shape[0]):
            if(image[j][i]==1.0):
                if(white_runs[m]>thresh):
                    img[j:j+white_runs[m],i]=1.0
                j+=white_runs[m]-1
                m+=1
            j+=1
    binary_img = img_as_ubyte(image)
    l=np.ones((1,80))
    vertical_shape = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.uint8(l))
    close = cv2.morphologyEx(vertical_shape, cv2.MORPH_CLOSE, np.uint8(l))

    start = 0
    for i in range(close.shape[0]):
        if (close[i][close.shape[1] // 2]):
            start = i
            break

    staff_lines = []    
    staff_line_place = []
    staff_group_height = 5 * (line_width + line_spacing) - line_spacing
    i = 0
    while(i <= staff_group_height):
        for j in range(line_width):
            staff_lines.append(start+j)
        i = i + line_width + line_spacing 
        start = start + line_width + line_spacing  

    for i in range (0,len(staff_lines),line_width):
        staff_line_place.append(sum(staff_lines[i: i + line_width]) / line_width)

    return img, staff_line_place
#################################
def remove_line_With_projection(binary_img, line_width, line_spacing):
    row_white_pixels_histogram = []
    staff_lines = []
    staff_line_place = []
    counters = []
    threshold = 0.4 * binary_img.shape[1]
    img_without_lines = binary_img.copy()

    for i in range(binary_img.shape[0]):
        num_white_pixels = 0
        for j in range(binary_img.shape[1]):
            if (binary_img[i][j] == 1):
                num_white_pixels += 1

            row_white_pixels_histogram.append(num_white_pixels)

        if(num_white_pixels > threshold):
            img_without_lines[i,:] = 0
            staff_lines.append(i)

    # get staff line places:
    counter = 0
    for i in range (0,len(staff_lines)-1):
        if(staff_lines[i+1]-staff_lines[i] == 1):
            counter = counter + 1
        else:
            counters.append(counter)
            counter = 0
        if(i == (len(staff_lines)-2)):
            counters.append(counter)

    j=0    
    i=0
    while (j < len(counters)):  
        if(counters[j] == 0):
            staff_line_place.append(staff_lines[i])
        else:
            staff_line_place.append(sum(staff_lines[i: i + counters[j] + 1]) / (counters[j]+1))
        i = i + counters[j] + 1
        j = j + 1

    arr = np.ones((line_width+2,line_width))
    dilation=binary_dilation(img_without_lines, selem=arr, out=None)
    close=binary_erosion(dilation, selem=arr, out=None) 

    return close, staff_line_place
###############################
def remove_line(binary_img, thresh, white_runs,line_width, line_spacing, rotating_angle):
# projection method for line removal
    if rotating_angle!=90.0:
        return remove_symbols(binary_img, thresh)
    else:
        return remove_line_With_projection(binary_img, line_width, line_spacing)
##################################
def split_lines(img,line_width, line_spacing):
    proj = np.sum(img, axis = 1)
    staff = np.where(proj > 0.5 * img.shape[0])[0]
    difs = np.diff(staff) 
    split = np.where(difs > 4 * line_spacing)[0]
    cuts = (staff[split] + staff[split + 1]) // 2
    lines = []
    prev_cut = 0
    for cut in cuts:
        lines.append(img[prev_cut:cut,:])
        prev_cut = cut
    lines.append(img[prev_cut:img.shape[0], :])
    return lines
################################
def Segment_Symbols(original_img, img,Line_Width,Line_Spacing):
    # find contours for symbols
    img=img.astype('uint8')
    arr=np.ones((Line_Spacing // 2,1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, arr)
    find_contour_img=find_contours(img,0.8)
    final= np.zeros(img.shape)
    # iterate on shapes
    Min_Ratio=(3*Line_Spacing+4*Line_Width)/(9*Line_Spacing)
    Max_Ratio=(5*Line_Spacing+5*Line_Width)/(Line_Spacing)
    arr=[]
    
    for box in find_contour_img:  
        Xmin=min(box[:,1])
        Xmax=max(box[:,1])
        Ymin=min(box[:,0])
        Ymax=max(box[:,0])
        ratio=(Ymax-Ymin)/(Xmax-Xmin)
        if ratio <=Max_Ratio and ratio>=Min_Ratio and (Xmax-Xmin)*(Ymax-Ymin)>2*Line_Width:
            rr, cc = rectangle(start = (math.ceil(0),math.ceil(Xmin)), end = (math.ceil(img.shape[0]),math.ceil(Xmax)), shape=img.shape)
            final[rr,cc]=1 
            arr.append([Xmin,Xmax,Ymin,Ymax])

    for element in arr:
        for box in find_contour_img: 
            Xmin=min(box[:,1])
            Xmax=max(box[:,1])
            Ymin=min(box[:,0])
            Ymax=max(box[:,0])
            if(element[0]>Xmin and element[1]<Xmax and element[2]>Ymin and element[3]<Ymax):
                arr.remove(element)
                break;
    arr.sort()
    segmented_shapes=[]
    segmented_shapes_long=[]
    for element in arr:
        segmented_shapes.append(original_img[int(element[2]): int(element[3]), int(element[0]): int(element[1])+2])
        
        segmented_shapes_long.append(original_img[0: original_img.shape[0], int(element[0]): int(element[1])])
    return segmented_shapes, segmented_shapes_long;
##############################
def count_contour(img,Line_Width,Line_Spacing,minArea):
    find_contour_img=find_contours(img,0.8)
    i=0
    for box in find_contour_img:  
        Xmin=min(box[:,1])
        Xmax=max(box[:,1])
        Ymin=min(box[:,0])
        Ymax=max(box[:,0])
        if (Ymax-Ymin)*(Xmax-Xmin)>minArea:
            i+=1
    return i
############################
def detectPrimitives(img_filled,img,line_width, line_spacing):
    img = img.astype('uint8')
    kernel_vertical_line=np.ones((2*line_spacing+2*line_width,1))
    line = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_vertical_line)

    kernel_horizontal_line=np.ones((1,3*line_spacing+2*line_width))
    line_horizontal = cv2.morphologyEx(img_as_ubyte(img), cv2.MORPH_HITMISS, kernel_horizontal_line)
    
    resha=line_spacing+4*line_width
    kernel_resha_line=np.zeros((resha,resha))
    kernel_resha_line[:,resha//2:(resha//2)+1]=1
    line_resha = cv2.morphologyEx(img_as_ubyte(img), cv2.MORPH_HITMISS, rotate(kernel_resha_line,45))
    count=count_contour(line_resha,line_width,line_spacing,line_width*line_width)
    
    disk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(line_spacing-line_width,line_spacing-line_width))
    disk_detection = cv2.morphologyEx(img_as_ubyte(img_filled), cv2.MORPH_HITMISS, img_as_ubyte(rotate(disk,45)))
    disk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(line_spacing-3*line_width,line_spacing-3*line_width))
    dilated_img = cv2.dilate(disk_detection, img_as_ubyte(rotate(disk,45)))


    return count
##############################
def fill(img):
    im_th = img.astype('uint8')
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out
#############################
def detect_circles(img,line_width,line_spacing):
    disk_detection = binary_opening(img, img_as_ubyte(disk(line_spacing//3)))

    hough_radii=line_spacing//2
    hough_res = hough_circle(disk_detection,hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res,[hough_radii,],min_xdistance=line_width*2, min_ydistance=line_spacing//2)

    return cx,cy
#################################
def line_for_segmentation(line, line_width, line_spacing,rotating_angle,white_runs):
    
    line=line.astype('uint8')
    top, bottom, left, right = [50]*4
    img_with_border = cv2.copyMakeBorder(line, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    img, staff_line_place = remove_line(img_with_border, line_width+line_width//2, white_runs,line_width, line_spacing,rotating_angle)
    img_filled=fill(img).astype('uint8')
    segmented_shapes, segmented_shapes_long = Segment_Symbols(img,img_filled,line_width,line_spacing)
    return segmented_shapes, segmented_shapes_long, staff_line_place
#################################
def get_ball_place(line_spacing, line_width, staff_line_place, cx, cy, i):
        num="1"
        t=""
        if (cy[i]<=staff_line_place[0]-line_spacing-2*line_width ):
            t=t+"b"
            num="2"
        elif(cy[i]<=staff_line_place[0]-line_spacing):
            t=t+"a"
            num="2"
        elif(cy[i]<=staff_line_place[0]-2*line_width):
            t=t+"g"
            num="2"   
        elif(cy[i]<=staff_line_place[0]+2*line_width and cy[i]>=staff_line_place[0]-2*line_width):
            t=t+"f"
            num="2"
        elif(cy[i]>staff_line_place[0]+line_width and cy[i]<staff_line_place[1]-line_width):
            t=t+"e"
            num="2"
        elif(cy[i]<=staff_line_place[1]+2*line_width and cy[i]>=staff_line_place[1]-2*line_width):
            t=t+"d"
            num="2"
        elif(cy[i]>=staff_line_place[1]+line_width and cy[i]<=staff_line_place[2]-line_width):
            t=t+"c"
            num="2"
        elif(cy[i]<=staff_line_place[2]+2*line_width and cy[i]>=staff_line_place[2]-2*line_width):
            t=t+"b"
        elif(cy[i]>staff_line_place[2]+line_width and cy[i]<staff_line_place[3]-line_width):
            t=t+"a"
        elif(cy[i]<=staff_line_place[3]+2*line_width and cy[i]>=staff_line_place[3]-2*line_width):
            t=t+"g"
        elif(cy[i]>staff_line_place[3]+line_width and cy[i]<staff_line_place[4]-line_width):
            t=t+"f"
        elif(cy[i]<staff_line_place[4]+2*line_width and cy[i]>staff_line_place[4]-2*line_width):
            t=t+"e"
        elif(cy[i]>staff_line_place[4] and cy[i]<=staff_line_place[4]+line_spacing):
            t=t+"d"
        else:
            t=t+"c"
        return t, num
##############################
def classify(line_width, line_spacing, segmented_shapes, segmented_shapes_long, classi, staff_line_place):
    my_type = []
    c=""
    length = 0
    for j in range(len(classi)):
        if(len(my_type) == 0 and classi[j] == 'barline'):
            my_type.append('\meter<"4/4">')
        if(classi[j] in ('clef','barline','natural')):
            continue
        elif(classi[j]=="sharp"):
            c="#"
        elif(classi[j]=="double_sharp"):
            c="##"
        elif(classi[j]=="flat"):
            c="&"
        elif(classi[j]=="double_flat"):
            c="&&"
        elif(classi[j]=="t_4_4"):
            my_type.append('\meter<"4/4">')
        elif(classi[j]=="t_4_2"):
            my_type.append('\meter<"4/2">')
        elif(classi[j]=="dot" and len(my_type) > 0):
            my_type[-1] += '.'
        else:
            d=""
            count = detectPrimitives(segmented_shapes_long[j], segmented_shapes[j], line_width, line_spacing)
            if classi[j]in('a_1','a_2','a_4'):
                    segmented_shapes_long[j]=fill(segmented_shapes_long[j]).astype('uint8')
            cx,cy=detect_circles(segmented_shapes_long[j],line_width,line_spacing)
            if(len(cy)!=0):
                cx,cy = zip(*sorted(zip(cx,cy)))
            if(classi[j][0]=="a" and count> 0 and classi[j][2:] in('8','16','32')):
                if(count==1):
                    d="/8"
                elif(count==2):
                    d="/16"
                elif(count>2):
                    d="/32"
                length=1
            elif(classi[j][0]=='a'):
                d="/"+classi[j][2:]
                length=1
            elif(classi[j][0]=="b"):
                length=int(classi[j][2:])//4
                d="/"+classi[j][2:]
            elif(classi[j]=="chord"):
                length=len(cy)
                d="/4"
            elif(classi[j]=="augmented"):
                length=1
                d="/4."
            arr=[]
            t="" 
            if(length>len(cy)):
                length=len(cy)
            for i in range(length):
                t1, num = get_ball_place(line_spacing, line_width, staff_line_place, cx, cy, i)
                t+=t1
                if(classi[j]=="chord"):
                    if(d in ('/8', '/16', '/32') and classi[j][0] != 'a'):
                        t=t+num+d+','
                    else:
                        t=t+c+num+d+","
                    arr.append(t)
                    t=""
                else:
                    if(d in ('/8', '/16', '/32') and classi[j][0] != 'a'):
                        t=t+num+d+" "
                    else:
                        t=t+c+num+d+" "
            arr.sort()
            if(classi[j]=="chord"):
                ch="{"
                for k in arr:
                    ch=ch+k
                ch = ch[:-1]
                ch=ch+"} " 
                t=ch
            t = t[:-1]
            if (t != ''):
                my_type.append(t)
            c=""
           
    return my_type
################################
def extract_hog_features(img):
    target_img_size = (64, 64)
    img = cv2.resize(img, target_img_size)
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()
###########################
def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model
##########################
def NN_classification(segmented_shapes, model):
    classi = []
    for img in segmented_shapes:
        img = img.astype('uint8')
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        featuresHog = extract_hog_features(img.astype('uint8'))
        classi.append(model.predict([featuresHog])[0])
    return classi
##########################
def process(path, model):
    img = io.imread(path, as_gray = True)
    io.imsave(Path(path).stem+'.PNG'+'img-read.PNG', img)
    img = local_binarize(img)
    img,rotating_angle = deskew(img)
    line_width, line_spacing, white_runs = find_ref_lengths(img)
    lines = split_lines(img, line_width, line_spacing)
    output_arr=[]
    for line in lines:
        segmented_shapes, segmented_shapes_long, staff_line_place=line_for_segmentation(line, line_width, line_spacing,rotating_angle,white_runs)
        classi = NN_classification(segmented_shapes, model)
        mytype = classify(line_width, line_spacing, segmented_shapes, segmented_shapes_long, classi, staff_line_place)
        output_arr.append(mytype)
    return output_arr
###############################
# argv
def OMR(input_file, output_file):
    model = load_model('NN_hog_model_pickle.sav')
    input_file += '/*'
    for img in glob.glob(input_file):
        f = open(output_file + '/' + Path(img).stem+'.txt', "w")  
        try:
            img_path=img
            output_arr = process(img_path, model)
            x="" 
            if len(output_arr)>1:
                x+='{\n'
            for i in range(len(output_arr)):
                output=output_arr[i]  
                x=x+'['
                for j in range(len(output)):
                    if(output[j]=='.'):
                        x=x+output[j]
                    else:
                        x=x+output[j]+" "   
                if(i==len(output_arr)-1):
                     x=x+"]\n"
                else:
                    x=x+"],\n"
            if len(output_arr)>1:
                x+='}'
            f.write(x)
            f.close() 
            print('Done image')   
        except:
            f.close()
            print("faild Successfully:)")
            continue

            

