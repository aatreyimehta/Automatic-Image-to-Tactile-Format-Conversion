#importing python packages to begin with
import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

#importing source data functions, channelwise target functions and grayscale target functions for 1024x512, 760x512 image dimensions from module bar_generator_horizontal2
from bar_generator_horizontal2 import generate_data, write_source_data_2, write_source_data_3, write_circle_target_data_2, write_circle_target_data1_2, write_circle_target_data_3, write_circle_target_data1_3, serialize_data

#importing postprocessing and maskgen from module utils
from utils import postprocessing, maskgen

NUM_SAMPLES = 1000   #samples generated per image dimension
P_grid = 0.8   #grid probability

os.makedirs("./data/source", exist_ok=True)   #folder to save source images
os.makedirs("./data/tactile", exist_ok=True)   #folder to save channelwise target images
os.makedirs("./data/tactile1", exist_ok=True)   #folder to save grayscale target images


with tf.device('/device:GPU:0'):
    data, metadata, circle_data = generate_data(num_samples=NUM_SAMPLES)
    
    

    for i in tqdm(range(len(data)), desc='bar charts: '):


        fig_size_2 = random.choices([[1024,512], [1024,512], [1024,512]], weights=[.5, .25, .25])[0]   #initializing figure size 1024x512(2:1 ratio)
        fig_size_3 = random.choices([[760,512], [760,512], [760,512]], weights=[.5, .25, .25])[0]   #initializing figure size 760x512(1.5:1 ratio)
        draw_grid = random.random() < P_grid
        tick_step = random.randint(10, 16)

        #calling write_source_data_2() from bar_generator_horizontal2 module
        write_source_data_2(data[i], f"./data/source/s_{i+4001}.png", fig_size_2, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+4001}.png")   #saves the images in source folder within data folder
        
        #calling write_target_data_2() from bar_generator_horizontal2 module
        write_circle_target_data_2(circle_data[i], f"./data/tactile/t_{i+4001}.png", fig_size_2, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+4001}.png")   #saves the images in tactile folder within data folder
        
        #calling write_target_data1_2() from bar_generator_horizontal2 module
        write_circle_target_data1_2(circle_data[i], f"./data/tactile1/t_{i+4001}.png", fig_size_2, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+4001}.png")   #saves the images in tactile folder within data folder
        
        #similarly, calling methods from bar_generator_horizontal2 module for 760x512 image dimensions
        write_source_data_3(data[i], f"./data/source/s_{i+2001}.png", fig_size_3, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+2001}.png")
        
        write_circle_target_data_3(circle_data[i], f"./data/tactile/t_{i+2001}.png", fig_size_3, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+2001}.png")
        
        write_circle_target_data1_3(circle_data[i], f"./data/tactile1/t_{i+2001}.png", fig_size_3, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+2001}.png")
        
        
        #write_circle_target_data2(circle_data[i], f"./data/tactile1/t_{i+5001}.png", fig_size1, draw_grid, tick_step)
        #print(f"./data/tactile1/t_{i+5001}.png")
        
    # metadata_df = serialize_data(metadata, ["x", "y"])
    # metadata_df.to_csv("metadata.csv", index=False)

# !zip -qq -r ./bardata.zip ./data/ ./metadata.csv
