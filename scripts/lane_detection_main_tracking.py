import numpy as np
import cv2
import time

from lane_util import warp_image, binary_pipeline, morphologic_process, get_poly_val, fit_track_lanes, \
    callback_ths, crop_points, draw_lane_img, visualize_images, visualize_images_angle, img_lane_tracking_point, img_lane_tracking_point_single, draw_tracking_point_lane_img, calculate_angle, draw_left_lane_img, draw_right_lane_img, draw_central_lane_img

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

def lane_detection(image):
    #initial threshold
    th_light = 95 
    
    cv2.namedWindow('Result')
    #cv2.createTrackbar('Luminance:', 'Result', th_light, 99, callback_ths)
    
    # lane function order
    poly_order = 1

    for _ in range(20000):

        # record time
        t_s = time.time()
  
        #resize image
        img = cv2.resize(image, dsize = (480,320))	
    
        img_binary = binary_pipeline(img,th_light, (80,95))
        img_binary_warp = warp_image(img_binary, source_points)
        img_morph = morphologic_process(img_binary_warp, 5, 1, 1)
        
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

            tracking_x, tracking_y = img_lane_tracking_point(left_fitx, left_fity, right_fitx, right_fity)
            base_x = img_line.shape[1]/2
            base_y = img_line.shape[0]
            img_tracking_point_lane = draw_tracking_point_lane_img(img_line,tracking_x,tracking_y,base_x,base_y)
            
            angle = calculate_angle(tracking_x,tracking_y,base_x,base_y)             
	elif fit_check==str('central_lane'):

            ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            central_fit = left_fit
            central_fitx = get_poly_val(ploty, central_fit)
            central_fitx, central_fity = crop_points(central_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
	    img_line = draw_central_lane_img(img_morph, central_fitx, central_fity)
            base_x = img_line.shape[1]/2
            base_y = img_line.shape[0] 
            angle,tracking_x,tracking_y = img_lane_tracking_point_single(central_fitx,central_fity,base_x)
            img_tracking_point_lane = draw_tracking_point_lane_img(img_line,tracking_x,tracking_y,base_x,base_y)
 
        elif fit_check ==str('left_lane'):
      	    ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            left_fitx = get_poly_val(ploty, left_fit)

            left_fitx, left_fity = crop_points(left_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
	    img_line = draw_left_lane_img(img_morph, left_fitx, left_fity)
            
            base_x = img_line.shape[1]/2
            base_y = img_line.shape[0] 
            angle,tracking_x,tracking_y = img_lane_tracking_point_single(left_fitx,left_fity,base_x)
            img_tracking_point_lane = draw_tracking_point_lane_img(img_line,tracking_x,tracking_y,base_x,base_y)
            
        
        elif fit_check ==str('right_lane'):
            ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            right_fitx = get_poly_val(ploty, right_fit)
            right_fitx, right_fity = crop_points(right_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
            img_line = draw_right_lane_img(img_morph, right_fitx, right_fity)
            
            base_x = img_line.shape[1]/2
            base_y = img_line.shape[0] 
            angle,tracking_x,tracking_y = img_lane_tracking_point_single(right_fitx,right_fity,base_x)
            img_tracking_point_lane = draw_tracking_point_lane_img(img_line,tracking_x,tracking_y,base_x,base_y)

        else:
            img_line = np.copy(img_morph)
            tracking_x = img_line.shape[1]/2
            tracking_y = img_line.shape[0]/2
            base_x = img_line.shape[1]/2
            base_y = img_line.shape[0]  
            img_tracking_point_lane = draw_tracking_point_lane_img(img_line,tracking_x,tracking_y,base_x,base_y)
            
            angle = calculate_angle(tracking_x,tracking_y,base_x,base_y) 
        print(fit_check)
        t_cost = time.time()-t_s

        th_light = visualize_images_angle([cv2.polylines(img,[source_points.astype(np.int32)],True,(255,0,0), 5),                                 
                                    img_tracking_point_lane],
                                    t_cost, params_img, angle, img_name=['src', 'lane'])
        return img_tracking_point_lane,angle
        
if __name__ == '__main__':
    lane_detection()
