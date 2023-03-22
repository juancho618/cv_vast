import cv2 as cv
import math
import numpy as np
from skimage.feature import hog
from skimage import exposure
import pickle

# add text to image
def add_text(frame, text, x, y, scale=1, color=(255, 0, 0), thickness=2, line_type=cv.LINE_AA):
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, text, (x, y), font, scale, color, thickness, line_type)
        return frame

videos_path = './videos'
video = cv.VideoCapture(f'{videos_path}/final.mp4')


###---- Grabbing
def find_contours(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 127, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def find_contours2(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    lower_red = np.array([0,25,130])
    upper_red = np.array([35,80,200])

    mask = cv.inRange(hsv, lower_red, upper_red)    
    inverted_mask = cv.bitwise_not(mask)


    # Erode and dilate the mask
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    mask_erode = cv.erode(inverted_mask, kernel1, iterations=1)
    mask_dilate = cv.dilate(inverted_mask, kernel2, iterations=1)

    # find contours in the mask 
    contours, hierarchy = cv.findContours(mask_dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, mask_erode, mask_dilate
  

####------

total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

fps = math.ceil(video.get(cv.CAP_PROP_FPS))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')
output_video = cv.VideoWriter('./videos/result/final3.mp4', fourcc, fps, (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT))))
four_sec_frames = int(fps * 4)
gray_frames = int(fps * 2)
output_video_face = cv.VideoWriter('./videos/result/static.mp4', fourcc, fps, (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT))))
###-- Edge detection
def find_edges_sobel_threshold(frame,threshold=22):
    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply Sobel operator
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude and direction of gradient
    mag, angle = cv.cartToPolar(sobelx, sobely, angleInDegrees=True)

    # Normalize magnitude for display
    mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    threshold = 22
    mag[mag < threshold] = 0
    mag[mag >= threshold] = 255
    return mag


def sobel_edge_detector(img):
    # Convert to graycsale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0) 

    # Sobel Edge Detection
    sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

    abs_sobelx = cv.convertScaleAbs(sobelx)
    abs_sobely = cv.convertScaleAbs(sobely)

    edges = cv.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    return edges

def canny_edge_detector(img):
    # Convert to graycsale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0) 

    # Canny Edge Detection
    edges = cv.Canny(image=img_blur, threshold1=90, threshold2=150) # Canny Edge Detection
    return edges

#-- end of edge detection

###-- Hough transform
def get_lines(edges, rho, theta, threshold):
        lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)
        return lines
def hough_transform_circle(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    circles = cv.HoughCircles(gray_blur, cv.HOUGH_GRADIENT, 1, 35, param1=12, param2=66, minRadius=55, maxRadius=150)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
                cv.circle(img, (x, y), r, (0, 0, 255), 2)
                cv.circle(img, (x, y), 2, (0, 255, 0), 3)
    return img

def hough_transform_line(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, threshold1=100, threshold2=200) # Canny Edge Detection        
    lines = []
    threshold = 60
    while len(lines) > 0:
            lines = get_lines(edges, 1, np.pi/180, threshold=50)
            threshold -= 5
    for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

def hough_transform_ellipsis(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(img_gray, (5, 5), 0)                

    # Detect edges using the Canny edge detector
    edges = cv.Canny(gray_blur, 50, 150)

    # Apply Hough transform to detect ellipses in the image
    ellipses = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 20,
                            param1=80, param2=37, minRadius=80, maxRadius=0)

    # Draw the detected ellipses on the input image
    if ellipses is not None:
            ellipses = np.round(ellipses[0, :]).astype("int")
            for (x, y, r) in ellipses:
                    cv.ellipse(img, (x, y), (r, r), 0, 0, 360, (0, 0, 255), 2)
            cv.imshow('Image', img)
            cv.waitKey(0)
    return img

#-- end Hough transform


###--- template matching
def template_matching(img, template_img):   

    # Convert both images to grayscale
    input_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)

    # Find the location of the template in the input image
    result = cv.matchTemplate(input_gray, template_gray, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Draw a rectangle around the location of the template in the input image
    h, w = template_gray.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    # Display the result
    return top_left, bottom_right,w, h
#-- end of template matching



###--- face and eyes detection
def eyes_face_detection(img):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")
    eyes = []
    roi_color = []

    # cv.imshow('edges_xy', img)
    # cv.waitKey(0)
    # Convert into grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        # Draw rectangles around the faces
        new_img = np.zeros_like(img)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

    return img,eyes, roi_color
#-- end face detection

###--- Extras

def sift(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()

    # Find keypoints and descriptors
    kp, des = sift.detectAndCompute(gray, None)
    return kp

def color_transfer(source, target):
    
    # Convert source and target images to Lab color space
    src_lab = cv.cvtColor(source, cv.COLOR_BGR2LAB)
    target_lab = cv.cvtColor(target, cv.COLOR_BGR2LAB)

    # Split source and target images into color channels
    src_l, src_a, src_b = cv.split(src_lab)
    target_l, target_a, target_b = cv.split(target_lab)

    source_l_mean, source_l_std = cv.meanStdDev(src_l)
    source_a_mean, source_a_std = cv.meanStdDev(src_a)
    source_b_mean, source_b_std = cv.meanStdDev(src_b)
    target_l_mean, target_l_std = cv.meanStdDev(target_l)
    target_a_mean, target_a_std = cv.meanStdDev(target_a)
    target_b_mean, target_b_std = cv.meanStdDev(target_b)

    target_l -= target_l_mean
    target_a -= target_a_mean
    target_b -= target_b_mean

    target_l = target_l_std / source_l_std * target_l
    target_a = target_a_std / source_a_std * target_a
    target_b = target_b_std / source_b_std * target_b

    target_l += source_l_mean
    target_a += source_a_mean
    target_b += source_b_mean

    result_lab = cv.merge((target_l, target_a, target_b))

    result_bgr = cv.cvtColor(result_lab, cv.COLOR_LAB2BGR)
    return result_bgr

def get_mean_and_std(x):
	x_mean, x_std = cv.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def color_transfer2(source, target):
    s_mean, s_std = get_mean_and_std(source)
    t_mean, t_std = get_mean_and_std(target)

    height, width, channel = source.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = source[i,j,k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
                x = round(x)
                # boundary check
                x = 0 if x<0 else x
                x = 255 if x>255 else x
                source[i,j,k] = x

    source = cv.cvtColor(source,cv.COLOR_LAB2BGR)
    return source
#-- end of extras

###--- probability
def get_histograms_from_images_gray(img):
   gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   hist = cv.calcHist([gray_img], [0], None, [256], [0, 256])
   return hist

def extract_crops_around_pixels(image, n):
    """
    Extracts a crop of size n x n around each pixel in an input image,
    with padding added if the crop goes outside the image boundaries.

    Args:
        image (numpy.ndarray): Input image
        n (int): Size of the crop to extract around each pixel

    Returns:
        crops (numpy.ndarray): Array of size (H, W, n, n, C) containing
                               the crops around each pixel in the input image
    """
    # Get the dimensions of the input image
    H, W, C = image.shape

    # Add padding around the input image to handle edge cases
    image_padded = cv.copyMakeBorder(image, n//2, n//2, n//2, n//2, cv.BORDER_CONSTANT, value=(0, 0, 0))

    # Initialize an array to store the crops around each pixel
    crops = np.zeros((H, W, n, n, C))

    # Loop over all pixels in the input image
    for i in range(n//2, H+n//2):
        for j in range(n//2, W+n//2):
            # Extract the crop around the current pixel
            crop = image_padded[i-n//2:i+n//2+1, j-n//2:j+n//2+1, :]
            # Save the crop to a separate file
            # filename = f'./crops/crop_{i}_{j}.png'
            # cv.imwrite(filename, crop)


            # Store the crop in the output array
            crops[i-n//2, j-n//2] = crop
        with open('crops.pickle', 'wb') as f:
            pickle.dump(crops, f)

    return crops

def calculate_mse_value(mean1, mean2):
    # Calculate the mean squared error between the two histograms
    mse = np.mean((mean1- mean2)**2)
    return mse
##-- end


current_frame = 0
sobel_frames=0
sobel_threshold = 60
video_exp= cv.VideoCapture(f'{videos_path}/result/static.mp4')
fps_exp = video_exp.get(cv.CAP_PROP_FPS)
while video.isOpened():
    ret, frame = video.read()
    if ret:
        # Modify the first 4 seconds of frames
        output_frame = frame
        if current_frame < four_sec_frames:
            # load object patch
            template_img_orange = cv.imread(f'{videos_path}/result/orange.png')
            template_img_orange = cv.resize(template_img_orange, (41, 41))
            object_patch_hist = get_histograms_from_images_gray(template_img_orange)
            mean_patch_hist = np.mean(object_patch_hist)
            # half size frame
            frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            crops = extract_crops_around_pixels(frame, 41)
            mse_array = np.zeros((crops.shape[0], crops.shape[1]))
            for i in range(0, crops.shape[0]):
                for j in range(0, crops.shape[0]):
                    # Extract the crop around the current pixel
                    crop = crops[i,j,:,:,:]        
                    crop = crop.reshape(41, 41, 3)
                    # img = np.uint8(crop)
                    # print(img.dtype)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    # crop_hist = get_histograms_from_images(img)
                    crop_hist = get_histograms_from_images_gray(img)
                    # crop_hist = get_hog_from_images(img)
                    crop_img_mean = np.mean(crop_hist)
                    crop_mse = calculate_mse_value(crop_img_mean, crop_img_mean)
                    mse_array[i,j] = crop_mse
            with open('mse_mean.pickle', 'wb') as f:
                pickle.dump(mse_array, f)

        else:
            output_frame = frame
        output_video.write(output_frame)
        current_frame += 1
    else:
        break

cv.destroyAllWindows()
video.release()
video_exp.release()
output_video.release()
# output_video_sobel.release()
output_video_face.release()