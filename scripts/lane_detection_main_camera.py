#!/usr/bin/env python
import numpy as np
import cv2
import time

from lane_util import warp_image, binary_pipeline, morphologic_process, get_poly_val, fit_track_lanes, \
    callback_ths, crop_points, draw_lane_img, visualize_images, draw_left_lane_img, draw_right_lane_img, draw_central_lane_img

params_img = {
    "WIDTH": 480, # image width
    "HEIGHT": 320, # image height
}

x = params_img["WIDTH"]
y = params_img["HEIGHT"]

'''source_points = np.float32([
    [0.05 * x, 0.75 * y],
    [(0.5 * x) - (x*0.13), (0.52)*y],
    [(0.5 * x) + (x*0.13), (0.52)*y],
    [x - (0.05 * x), 0.75 * y]
    ])'''

source_points = np.float32([
    [0.1 * x, 0.95 * y],
    [(0.5 * x) - (x*0.35), (0.5)*y],
    [(0.5 * x) + (x*0.35), (0.5)*y],
    [x - (0.1 * x), 0.95 * y]
    ])
print(source_points)

#initial threshold
th_light = 75
hue_l = 80
hue_h = 95 
s_x1 = 48
s_x2 = 72
s_y1 = 304
s_y2 = 160
    
cv2.namedWindow('Result')
cv2.createTrackbar('Luminance:', 'Result', th_light, 99, callback_ths)
cv2.createTrackbar('Hue_low:', 'Result', hue_l, 255, callback_ths)
cv2.createTrackbar('Hue_high:', 'Result', hue_h, 255, callback_ths) 
cv2.createTrackbar('source_x1:', 'Result', s_x1, 240, callback_ths)
cv2.createTrackbar('source_x2:', 'Result', s_x2, 240, callback_ths)
cv2.createTrackbar('source_y1:', 'Result', s_y1, 360, callback_ths)
cv2.createTrackbar('source_y2:', 'Result', s_y2, 360, callback_ths)


def lane_detection(image):
    
    # lane function order
    poly_order = 1
    

    for _ in range(20000):

        # record time
        t_s = time.time()
        
	th_light = cv2.getTrackbarPos('Luminance:','Result')
  	hue_l = cv2.getTrackbarPos('Hue_low:','Result')
	hue_h = cv2.getTrackbarPos('Hue_high:','Result')
	s_x1 = cv2.getTrackbarPos('source_x1:','Result')
	s_x2 = cv2.getTrackbarPos('source_x2:','Result')
	s_y1 = cv2.getTrackbarPos('source_y1:','Result')
	s_y2 = cv2.getTrackbarPos('source_y2:','Result')

        #resize image
        img = cv2.resize(image, dsize = (480,320))
		
    
        img_binary = binary_pipeline(img,th_light, (hue_l,hue_h))
	source_points = np.float32([
    [s_x1, s_y1],
    [s_x2, s_y2],
    [x-s_x2, s_y2],
    [x-s_x1, s_y1]
    ])	
        img_binary_warp = warp_image(img_binary, source_points)
        img_morph = morphologic_process(img_binary_warp, 5, 3, 1)
        
        #t_cost = time.time()-t_s
    
        fit_check, left_fit, right_fit = fit_track_lanes(img_morph, poly_order=poly_order,
                                                                    nwindows=30,
                                                                    margin=30,
                                                                    minpix=20)

        img_morph = cv2.cvtColor(img_morph*255, cv2.COLOR_GRAY2BGR)

        if fit_check==str('two_lanes'):
            ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            left_fitx = get_poly_val(ploty, left_fit)
            right_fitx = get_poly_val(ploty, right_fit)

            left_fitx, left_fity = crop_points(left_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
            right_fitx, right_fity = crop_points(right_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])     

            img_line = draw_lane_img(img_morph, left_fitx, left_fity, right_fitx, right_fity)
           
	elif fit_check==str('central_lane'):

            ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            central_fit = left_fit
            central_fitx = get_poly_val(ploty, central_fit)
            central_fitx, central_fity = crop_points(central_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
	    img_line = draw_central_lane_img(img_morph, central_fitx, central_fity)

        elif fit_check ==str('left_lane'):
      	    ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            left_fitx = get_poly_val(ploty, left_fit)

            left_fitx, left_fity = crop_points(left_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
	    img_line = draw_left_lane_img(img_morph, left_fitx, left_fity)
        
        elif fit_check ==str('right_lane'):
            ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            right_fitx = get_poly_val(ploty, right_fit)
            right_fitx, right_fity = crop_points(right_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
        
            img_line = draw_right_lane_img(img_morph, right_fitx, right_fity)

        else:
            img_line = np.copy(img_morph)

        print(fit_check)
        t_cost = time.time()-t_s

        th_light = visualize_images([cv2.polylines(img,[source_points.astype(np.int32)],True,(255,0,0), 5),                                 
                                    img_line],
                                    t_cost, params_img, img_name=['src', 'lane'])
        return img_line
        
if __name__ == '__main__':
    lane_detection()
