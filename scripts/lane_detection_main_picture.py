#!/usr/bin/env python
import numpy as np
import cv2
import time

from lane_util_picture import warp_image, binary_pipeline, morphologic_process, get_poly_val, fit_track_lanes, \
    callback_ths, crop_points, draw_lane_img, visualize_images_simple

params_img = {
    "WIDTH": 480, # image width
    "HEIGHT": 320, # image height
}

x = params_img["WIDTH"]
y = params_img["HEIGHT"]

source_points = np.float32([
    [0.01 * x, 0.9 * y],
    [(0.5 * x) - (x*0.09), (0.52)*y],
    [(0.5 * x) + (x*0.09), (0.52)*y],
    [x - (0.01 * x), 0.9 * y]
    ])

def lane_detection():
    #initial threshold
    th_light = 75 
    
    cv2.namedWindow('Result')
    cv2.createTrackbar('Luminance:', 'Result', th_light, 99, callback_ths)
    
    # lane function order
    poly_order = 1

    for _ in range(20000):

        # record time
        t_s = time.time()

        image = cv2.imread('./image_sample/image_straight.jpg')
  
        #resize image
        img = cv2.resize(image, dsize = (480,320))	
    
        img_binary = binary_pipeline(img,th_light, (80,95))
        img_binary_warp = warp_image(img_binary, source_points)
        img_morph = morphologic_process(img_binary_warp, 5, 1, 1)
        
        t_cost = time.time()-t_s
    
        fit_check, left_fit, right_fit = fit_track_lanes(img_morph, poly_order=poly_order,
                                                                    nwindows=30,
                                                                    margin=30,
                                                                    minpix=20)

        img_morph = cv2.cvtColor(img_morph*255, cv2.COLOR_GRAY2BGR)

        if fit_check:

            ploty = np.linspace(0, params_img["HEIGHT"]-1, params_img["HEIGHT"])
            left_fitx = get_poly_val(ploty, left_fit)
            right_fitx = get_poly_val(ploty, right_fit)

            left_fitx, left_fity = crop_points(left_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
            right_fitx, right_fity = crop_points(right_fitx.astype(np.int32), ploty.astype(np.int32), params_img["WIDTH"], params_img["HEIGHT"])
        
            img_line = draw_lane_img(img_morph, left_fitx, left_fity, right_fitx, right_fity)
                      

        else:
            img_line = np.copy(img_morph)
            
        th_light = visualize_images_simple([cv2.polylines(img,[source_points.astype(np.int32)],True,(255,0,0), 5),
                                    cv2.cvtColor(img_binary*255, cv2.COLOR_GRAY2BGR),
                                    img_morph,
                                    img_line],
                                    t_cost,params_img, img_name=['src', 'binary', 'morph', 'lane'])
        
if __name__ == '__main__':
    lane_detection()
