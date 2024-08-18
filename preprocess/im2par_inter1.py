import os 
import cv2
import numpy as np 
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

def convert_to_np(image, resolution):
    image = np.frombuffer(image, dtype = np.uint8)
    image = image.reshape(resolution,resolution,3)
    return image

parquet_folder_path = '/home1/parquet'
folder_path = '/home1/grid_images/inter1'

frames_condition = os.listdir(f'{folder_path}/condition')
frames_input = os.listdir(f'{folder_path}/input')
frames_output = os.listdir(f'{folder_path}/output')
frames = list(set(frames_condition).intersection(frames_input))
frames = list(set(frames).intersection(frames_output))
print(len(frames))
frames.sort()

csv_file = "/home1/results_2M_train.csv"
df_results = pd.read_csv(csv_file)

order = 1
print(order * len(frames)//2, (order + 1) * len(frames)//2)

for idx_ in tqdm.tqdm(range(order * len(frames)//2, (order + 1) * len(frames)//2, 500)):
    data = []
    index = idx_ // 500
    
    if os.path.isfile(f'{parquet_folder_path}/sd15_256_2x2_inter1_{index}.parquet'):
        print('file exist')
        continue
    
    for frame in tqdm.tqdm(frames[idx_:idx_ + 500]):
        try:
            video_id = frame.split('_')[-1].split('.')[0]
            text = "Fill in the blanks, " + df_results[df_results['videoid'] == int(video_id)]['name'].values[0]

            frame_condition = cv2.imread(f'{folder_path}/condition/{frame}')
            frame_input = cv2.imread(f'{folder_path}/input/{frame}')
            frame_output = cv2.imread(f'{folder_path}/output/{frame}')
            
            frame_condition = cv2.cvtColor(frame_condition, cv2.COLOR_BGR2RGB)
            frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)

            condition_frame_bytes = frame_condition.tobytes()
            input_frame_bytes = frame_input.tobytes()
            output_frame_bytes = frame_output.tobytes()
            prompt = text
        
            data.append([input_frame_bytes,condition_frame_bytes, prompt, output_frame_bytes])
        except: 
            print('error')

    # original_image, condition_image, edit_image, edited_image_column
    df = pd.DataFrame(data, columns=['input_image','condition_image', 'edit_prompt', 'output_image'])
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'{parquet_folder_path}/sd15_256_2x2_inter1_{index}.parquet')