# LANE DETECTION PROGRAM
import cv2
import numpy as np

def make_coordinates(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*(3/5))         
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):

    left_fitting = []

    right_fitting = []

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        parameters = np.polyfit((x1, x2), (y1, y2), 1)

        slope = parameters[0]

        intercept = parameters[1]

        if slope < 0:

            left_fitting.append((slope, intercept))
            

        else:

            right_fitting.append((slope, intercept))
            

    left_fitting_average = np.average(left_fitting, axis=0)

    right_fitting_average = np.average(right_fitting, axis=0)
    

    try:

        left_line = make_coordinates(image, left_fitting_average)

        right_line = make_coordinates(image, right_fitting_average)

        return np.array([left_line, right_line])
     
    except Exception as e:

      print ('\n')
        
       

 

def canny_func(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = 5

    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def display_lines(image,lines):

    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image



def region_of_interest(canny):
    
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# commented out code can be used for single image lane detection

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# lane_canny = canny(lane_image)
# cropped_canny = region_of_interest(lane_canny)

# lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
# averaged_lines = average_slope_intercept(image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)


captureframes = cv2.VideoCapture("trialvideo.mp4")

while(captureframes.isOpened()):

     _, frame = captureframes.read()
     canny_image = canny_func(frame)
     cropped_canny = region_of_interest(canny_image)
     lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)

     averaged_lines = average_slope_intercept(frame, lines)
     line_image = display_lines(frame, averaged_lines)
     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
     cv2.imshow("result", combo_image)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

captureframes.release()
cv2.destroyAllWindows()        

 
