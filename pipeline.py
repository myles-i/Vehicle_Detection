import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
exec(open('helper_functions.py').read())
import pdb

(svc, X_scaler, color_space , orient, pix_per_cell, cell_per_block, hog_channel, 
    spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat) = pickle.load( open("svc_pickle.p", "rb" ) )

image = mpimg.imread('test_images/test1.jpg')
# image = mpimg.imread('bbox-example-image.jpg')
    

## generate list if windows for detection
center = 475
# xy_window_list    = [(426,426), (320,320), (256,256), (128,128), (64,64)]
# x_start_stop_list = [[0, 1280], [0, 1280], [0, 1280], [200, 1180], [400, 1280]]
xy_window_list    = [(200,200), (100,100), (60,60)]
x_start_stop_list = [[380, 1280], [480, 1280], [620, 1280]]
y_start_stop_list = [[300, 650], [350, 600], [375, 500]]
for xy_window in xy_window_list:
    half_height = xy_window[0]//2
    y_start_stop_list.append([center - np.int32(half_height*2), center + np.int32(half_height*2)])
windows = []
for y_start_stop, x_start_stop, xy_window in zip(y_start_stop_list, x_start_stop_list, xy_window_list):
    windows = windows + slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=xy_window, xy_overlap=(0.5, 0.5))
# window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)    
# plt.figure( )
# plt.imshow(window_img)
# plt.show(block=False)

heat = np.zeros_like(image[:,:,0]).astype(np.float)
init = True
def detection_pipeline(image, show_result = False, alpha = 0.8):
    # image = mpimg.imread('bbox-example-image.jpg')
    draw_image = np.copy(image)
    # ystart = 400
    # ystop  = 656
    # xstart = 400
    # xstop  = 1280
    # scales            = [3,           2,           1.5,          1,           .5]
    # x_start_stop_list = [[600, 1280], [384, 1280], [480, 1280], [624, 1024], [624, 896]]
    # y_start_stop_list = [[350, 656] , [400, 656],  [400, 600],  [400, 480],  [400, 450]]

    scales            = [2,           1.5,          1,           .5]
    x_start_stop_list = [[384, 1280], [480, 1280], [624, 1024], [624, 896]]
    y_start_stop_list = [[400, 656],  [400, 600],  [400, 480],  [400, 450]]

    hot_windows = []
    all_windows = []
    for y_start_stop, x_start_stop, scale in zip(y_start_stop_list, x_start_stop_list, scales):

        new_hot_windows, new_all_windows = find_cars(image, y_start_stop[0], y_start_stop[1], x_start_stop[0], x_start_stop[1], scale, svc,
         X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat, color_space = color_space)
        hot_windows = hot_windows + new_hot_windows
        all_windows = all_windows + new_all_windows
    # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
    #                             spatial_size=spatial_size, hist_bins=hist_bins, 
    #                             orient=orient, pix_per_cell=pix_per_cell, 
    #                             cell_per_block=cell_per_block, 
    #                             hog_channel=hog_channel, spatial_feat=spatial_feat, 
    #                             hist_feat=hist_feat, hog_feat=hog_feat) 
    # Add heat to each box in box list


    # window_img = draw_boxes(image, all_windows, color=(0, 0, 255), thick=6)    
    # plt.figure( )
    # plt.imshow(window_img)
    # plt.show(block=False)
    global heat
    heat = add_heat(heat,hot_windows, alpha)
        
    # Apply threshold to help remove false positives
    heat_thresh = apply_threshold(heat.copy(),1.5)

    # Visualize the heatmap when displayibng    
    heatmap = np.clip(heat_thresh, 0, 255)
    print(np.max(heatmap))

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels)
    if show_result:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heat, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show(block = False) #block = False
    return draw_img



from moviepy.editor import VideoFileClip
def test_on_video(vid_name):
    output_name = "output_" + vid_name
    clip1 = VideoFileClip(vid_name)
    vid_clip = clip1.fl_image(detection_pipeline) #NOTE: this function expects color images!!
    vid_clip.write_videofile(output_name, audio=False)
# pdb.set_trace()
# image = mpimg.imread('test_images/test1.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)
# image = mpimg.imread('test_images/test1.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)
# image = mpimg.imread('test_images/test2.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)
# image = mpimg.imread('test_images/test3.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)
# image = mpimg.imread('test_images/test4.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)
# image = mpimg.imread('test_images/test5.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)
# image = mpimg.imread('test_images/test6.jpg')
# init = True
# detection_pipeline(image, show_result = True, alpha = 0.)


# test_on_video('test_video.mp4')
test_on_video('project_video.mp4')

# ystart = 400
# ystop = 656
# scale = 2
    
# out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,spatial_feat=spatial_feat, 
#                                 hist_feat=hist_feat, hog_feat=hog_feat, color_space = color_space)

# plt.imshow(out_img)
# plt.show(block = False)#block = False