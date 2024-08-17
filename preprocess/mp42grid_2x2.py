import cv2
from PIL import Image
import tqdm 
import glob
import numpy as np
import os
import pandas as pd 
import torch 
import random

def crop_2x2(frame):
    if frame == 0:
        return 0,0,254,254
    elif frame == 1:
        return 258,0,258+254,254
    elif frame == 2:
        return 0,258,254,254+258
    elif frame == 3:
        return 258,258,258+254,258+254  

def get_merged_image(frame_lists):
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr.fill(0)
    result = Image.fromarray(arr)
    result.paste(Image.fromarray(frame_lists[0]), (0, 0))
    result.paste(Image.fromarray(frame_lists[1]), (258, 0))
    result.paste(Image.fromarray(frame_lists[2]), (0, 258))
    result.paste(Image.fromarray(frame_lists[3]), (258, 258))
    return result

def get_masked_image(original_image, masked_list=[3,-1,-1,-1], f_idx=0):
    result = np.array(Image.new("RGB", (512, 512)))

    original_image_np = np.array(original_image)
    for new_img_idx, mask_idx in enumerate(masked_list):
        n_x1, n_y1, n_x2, n_y2 = crop_2x2(new_img_idx) 

        if mask_idx == -1:
            result[n_y1:n_y2, n_x1:n_x2] = np.full((254, 254, 3), 255, dtype = np.uint8)
        else:
            x1,y1,x2,y2 = crop_2x2(mask_idx)
            result[n_y1:n_y2, n_x1:n_x2] = original_image_np[y1:y2, x1:x2]
    result = Image.fromarray(result)

    return result

def center_crop(image, target_size):
    h, w = image.shape[:2]
    target_width, target_height = target_size
    center_x = w // 2
    center_y = h // 2
    crop_half_width = target_width // 2
    crop_half_height = target_height // 2
    crop_left = max(center_x - crop_half_width, 0)
    crop_top = max(center_y - crop_half_height, 0)
    crop_right = min(center_x + crop_half_width, w)
    crop_bottom = min(center_y + crop_half_height, h)
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
    return cropped_image

def index2image(frame_indexs, blank = True, masked_list = [-1,-1,-1,-1]):
    image_frames = []
    for frame_index in frame_indexs:
        image_frames.append(frames[frame_index])
    image = get_merged_image(image_frames)
    if blank:
        image = get_masked_image(image, masked_list)
    return image

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def filein(path, v_name):
    create_directory_if_not_exists(f'{path}/t2v')
    create_directory_if_not_exists(f'{path}/key/output')
    create_directory_if_not_exists(f'{path}/key/condition')
    create_directory_if_not_exists(f'{path}/inter1/input')
    create_directory_if_not_exists(f'{path}/inter1/output')
    create_directory_if_not_exists(f'{path}/inter1/condition')
    create_directory_if_not_exists(f'{path}/inter2/input')
    create_directory_if_not_exists(f'{path}/inter2/output')
    create_directory_if_not_exists(f'{path}/inter2/condition')

    if (os.path.isfile(f'{path}/t2v/{v_name}.png') and
        os.path.isfile(f'{path}/key/output/{v_name}.png') and 
        os.path.isfile(f'{path}/key/condition/{v_name}.png') and
        os.path.isfile(f'{path}/inter1/input/{v_name}.png') and 
        os.path.isfile(f'{path}/inter1/output/{v_name}.png') and 
        os.path.isfile(f'{path}/inter1/condition/{v_name}.png') and 
        os.path.isfile(f'{path}/inter2/input/{v_name}.png') and 
        os.path.isfile(f'{path}/inter2/output/{v_name}.png') and 
        os.path.isfile(f'{path}/inter2/condition/{v_name}.png')): return True
    
    else: return False


output_folder_path = "/home1/s20225518/T2V_CVPR_2024/test"
csv_file = "/home2/taegyeong/results_2M_train.csv"
df_results = pd.read_csv(csv_file)

split_number = 100
order = 12
video_folder_paths = "/data6/webvid_backup/*"

folder_list = sorted(glob.glob(video_folder_paths))[order * split_number: (order + 1) * split_number]

for folder in tqdm.tqdm(folder_list):
    for v_path in tqdm.tqdm(glob.glob('{}/*.mp4'.format(folder))):
        try:
            video = cv2.VideoCapture(v_path) 
            v_name = v_path.split('/')[-2] + '_' +  v_path.split('/')[-1][:-4]
            video_id = v_path.split('/')[-1][:-4]

            if filein(output_folder_path, v_name): 
                print('file exists')
                continue
            
            fps = round(video.get(cv2.CAP_PROP_FPS))
            
            # all frames
            frames = []
            success = True
            while success:
                success, frame = video.read()
                if not success: break

                height = 254
                aspect_ratio = float(height) / frame.shape[0]
                dsize = (int(frame.shape[1] * aspect_ratio), height)
                
                frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
                frame = center_crop(frame, (254,254))

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                frames.append(frame)
        
            video.release()
                
            total_frames = len(frames)
            
            if total_frames < (fps * 21):
                continue
            
            random_frame_index_list = random_integer = random.sample([i for i in range(0, total_frames)], 10)
            
            # t2v The interval between each inside frame is 3 seconds.
            t2v_frames_index = [i * fps for i in range(0,22,3)] # 0,3,6,9,12,15,18,21
            t2v_merged_image = index2image(t2v_frames_index[:4], blank=False)
            t2v_merged_image.save(f'{output_folder_path}/t2v/{v_name}.png')
            
            key_frames_index_condition = t2v_frames_index[:4]
            key_frames_index = t2v_frames_index[4:]  # for next key grid image generation models
            
            key_merged_image_condition = index2image(key_frames_index_condition, blank=False)
            key_merged_image_condition.save(f'{output_folder_path}/key/condition/{v_name}.png')
            key_merged_image = index2image(key_frames_index, blank=False)
            key_merged_image.save(f'{output_folder_path}/key/output/{v_name}.png')
            
            # interpolation 1 step, The interval between each inside frame is 1 seconds.
            inter1_step = 1
            end = key_frames_index_condition[1]

            inter1_frames_index_condition = [round(round(i * inter1_step, 2) * fps) for i in range(4)] 
            inter1_frames_index = [end + round(round(i * inter1_step, 2) * fps) for i in range(4)] 
            
            inter1_merged_image_condition = index2image(inter1_frames_index_condition, blank=False)
            inter1_merged_image_condition.save(f'{output_folder_path}/inter1/condition/{v_name}.png')
            
            inter1_merged_image_output = index2image(inter1_frames_index, blank=False)
            inter1_merged_image_input = index2image(inter1_frames_index, blank=True, masked_list=[0,-1,-1,3])
            
            inter1_merged_image_output.save(f'{output_folder_path}/inter1/output/{v_name}.png')
            inter1_merged_image_input.save(f'{output_folder_path}/inter1/input/{v_name}.png')

            # interpolation 2 step, The interval between each inside frame is 1 seconds.
            inter2_step = 0.33
            end = inter1_frames_index_condition[1]
            
            inter2_frames_index_condition = [round(round(i * inter2_step, 2) * fps) for i in range(4)] 
            inter2_frames_index = [end + round(round(i * inter2_step, 2) * fps) for i in range(4)] 
            
            inter2_merged_image_condition = index2image(inter2_frames_index_condition, blank=False)
            inter2_merged_image_condition.save(f'{output_folder_path}/inter2/condition/{v_name}.png')
            
            inter2_merged_image_output = index2image(inter2_frames_index, blank=False)
            inter2_merged_image_input = index2image(inter2_frames_index, blank=True, masked_list=[0,-1,-1,3])
            
            inter2_merged_image_output.save(f'{output_folder_path}/inter2/output/{v_name}.png')
            inter2_merged_image_input.save(f'{output_folder_path}/inter2/input/{v_name}.png')

        except:
             print(v_path, total_frames, fps)
