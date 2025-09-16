import numpy as np
import cv2
import uuid
import os

GRAP_LINE = 15

def Gaussian_kernel(sigma, truncate=1):

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]

    sigma = sigma**2

    k = np.exp((x**2 + y**2)/(2*sigma))*2
    k = k/np.sum(k)

    return k

def liner_regression(data): #data [(x1, y1), (x2, y2), (x3, y3)...]
    #[a, b] = (A_transpo * A)^-1 * A_transpo * y
    #with A = [(x1, 1), (x2, 1), (x3, 1), ...]
    #      y = [y1, y2, y3....]

    A = np.array([(x, 1) for x, y in data], dtype=float)   # n x 2
    y = np.array([y for x, y in data], dtype=float).reshape(-1, 1)  # n x 1 column

    # Compute [a, b] = (A^T A)^-1 A^T y
    A_T = A.T
    inv = np.linalg.inv(A_T @ A)
    result = inv @ A_T @ y

    return result.flatten()  # returns [a, b]




def border_detect(binary_image):
    h, w = binary_image.shape
    boder_y = []

    for x in range(w):
        col = binary_image[:, x]

        idx = np.where((col[:-1]==255) & (col[1:] == 0))[-1]

        if(len(idx)>0):
            y = idx[-1]
            boder_y.append((x, y))


    line = liner_regression(boder_y)
    return line







def lane_detect(image, key):

    h, w = image.shape

    kernel = Gaussian_kernel(2)
    blured = cv2.filter2D(image, -1, kernel)


    #delete sky
    _, delete_sky = cv2.threshold(blured, 100, 255, cv2.THRESH_BINARY)
    a, b = border_detect(delete_sky)
    b += GRAP_LINE
    
    
    for x in range(w):
        y = int(a*x + b)
        
                

        if(y<0):
            y = 0
        if(y>=h):
            y = h-1

        blured[:y, x] = 0
        
    #delete ground
    _, delete_ground = cv2.threshold(blured, 50, 255, cv2.THRESH_BINARY)

    curent = 0
    max_length = 0
    best_start = -1
    curent_start = -1



    for x in range(w):
        y = int(a*x + b)
        if(0 <= y < h):
            if(delete_ground[y,x]==255):
                if(curent==0):
                    curent_start = x
                curent+=1

                if(curent>max_length):
                    max_length = curent
                    best_start = curent_start
                
            else:
                curent = 0

    y = int(a * curent_start + b)
    top_Left = (curent_start, y)


    right_x = min(w-1, curent_start + max_length)
    y = int(a * right_x + b)
    top_Right = (right_x, y)


    cv2.imshow("gray_image", blured)
    

    if(key==ord("t")):
        folder = "image"
        os.makedirs(folder, exist_ok=True)
        random_name = str(uuid.uuid4()) + ".png"

        save_path = os.path.join(folder,random_name)

        cv2.imwrite(save_path, delete_ground)


    cv2.imshow("delete_ground", delete_ground)
    
    return (top_Left, top_Right)
