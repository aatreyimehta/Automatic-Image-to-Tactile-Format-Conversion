#importing python packages to begin with
import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

#importing source data functions, channelwise target functions and grayscale target functions for 512x512, 512x1024, 512x700 image dimensions from module bar_generator_horizontal
from bar_generator_horizontal import generate_data, write_source_data, write_source_data_elongated, write_circle_target_data, write_circle_target_data1, write_circle_target_data_elongated, write_circle_target_data1_elongated, write_source_data_1, write_circle_target_data_1, write_circle_target_data1_1,  serialize_data

#importing postprocessing and maskgen from module utils
from utils import postprocessing, maskgen

NUM_SAMPLES = 1000  #samples generated per image dimension
P_grid = 0.8    #grid probability

os.makedirs("./data/source", exist_ok=True)   #folder to save source images
os.makedirs("./data/tactile", exist_ok=True)  #folder to save channelwise target images
os.makedirs("./data/tactile1", exist_ok=True) #folder to save grayscale target images


with tf.device('/device:GPU:0'):
    data, metadata, circle_data = generate_data(num_samples=NUM_SAMPLES)
    
    

    for i in tqdm(range(len(data)), desc='bar charts: '):

        fig_size = [512,512]  #initializing figure size 512x512(1:1 ratio)
        fig_size_elongated = random.choices([[512,1024], [512,1024], [512,1024]], weights=[.5, .25, .25])[0]   #initializing figure size 512x1024(1:2 ratio)
        fig_size_1 = random.choices([[512,700], [512,700], [512,700]], weights=[.25, .5, .5])[0]     #initializing figure size 512x700(1:1.5 ratio)

        draw_grid = random.random() < P_grid
        tick_step = random.randint(10, 16)

        #calling write_source_data() from bar_generator_horizontal module
        write_source_data(data[i], f"./data/source/s_{i+1}.png", fig_size, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+1}.png")  #saves the images in source folder within data folder
        
        #calling write_target_data() from bar_generator_horizontal module
        write_circle_target_data(circle_data[i], f"./data/tactile/t_{i+1}.png", fig_size, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+1}.png")   #saves the images in tactile folder within data folder
        
        #calling write_target_data1() from bar_generator_horizontal module
        write_circle_target_data1(circle_data[i], f"./data/tactile1/t_{i+1}.png", fig_size, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+1}.png")  #saves the images in tactile1 folder within data folder
        
        #similarly, calling methods from bar_generator_horizontal module for other two image dimensions
        write_source_data_elongated(data[i], f"./data/source/s_{i+1001}.png", fig_size_elongated, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+1001}.png")
        
        write_circle_target_data_elongated(circle_data[i], f"./data/tactile/t_{i+1001}.png", fig_size_elongated, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+1001}.png")
        
        write_circle_target_data1_elongated(circle_data[i], f"./data/tactile1/t_{i+1001}.png", fig_size_elongated, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+1001}.png")
        
        write_source_data_1(data[i], f"./data/source/s_{i+3001}.png", fig_size_1, draw_grid, tick_step)
        postprocessing(f"./data/source/s_{i+3001}.png")
        
        write_circle_target_data_1(circle_data[i], f"./data/tactile/t_{i+3001}.png", fig_size_1, draw_grid, tick_step)
        maskgen(f"./data/tactile/t_{i+3001}.png")
        
        write_circle_target_data1_1(circle_data[i], f"./data/tactile1/t_{i+3001}.png", fig_size_1, draw_grid, tick_step)
        print(f"./data/tactile1/t_{i+3001}.png")
        
        
    # metadata_df = serialize_data(metadata, ["x", "y"])
    # metadata_df.to_csv("metadata.csv", index=False)

# !zip -qq -r ./bardata.zip ./data/ ./metadata.csv
