#importing python packages 
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from generators.generators import create_gen
from datasets.datasets import get_dataset
from util import mkdir

#Opt takes a dictionary as input and converts its key-value pairs to object attributes
class Opt:
     def __init__(self, dictionary):
        for k, v in dictionary.items():
             setattr(self, k, v)

#load_opt  loads a JSON file from the specified path and returns an Opt instance
def load_opt(path):
    with open(path) as json_file:
        opt = json.load(json_file)
    
    opt = Opt(opt)
    return opt

#load_model loads a generator model from the specified path and returns it
def load_model(model_path,opt,device):
    G = create_gen(opt.gen,opt.input_dim,opt.output_dim,opt.gen_filters,opt.norm,multigpu=False)
    G.to(device)
    
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint["gen"], strict=False)
    return G

def load_data(photo_path,opt):
    data = get_dataset(photo_path, opt, mode='test')
    dataset = DataLoader(dataset=data, batch_size=1, shuffle=False,num_workers=4)
    return dataset

def unnormalize(a):  #unnormalization of input tensor
    return a/2 +0.5

def concat_images(photo,sketch,output):

    return torch.cat((photo,sketch,output),2)


def save_images(dataset,path):
    for i, batch in enumerate(dataset):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = Gen(real_A.to(device))[0].cpu()
            
        a = unnormalize(real_A[0])   #unnormalized source image
        b = unnormalize(real_B[0])   #unnormalized target image
        out = unnormalize(out)      #unnormalized output image
        
        file_name = str(i+1) +".png"   #filename of the saved images
        save_image(concat_images(a,b,out), os.path.join(path,file_name))   #horizontally concatenated source, target and trained output images
        print(f"file saved at: {os.path.join(path,file_name)}")

opt_path = os.path.join(os.getcwd(),"models","pix2pix","params.txt")   #output folder 
opt = load_opt(opt_path)
device = torch.device("cuda:0")

model_path = os.path.join(os.getcwd(),"models",opt.folder_load,"final_model.pth")
Gen = load_model(model_path,opt,device)

photo_path_test = os.path.join(os.getcwd(),"data","test", "source")
dataset = load_data(photo_path_test,opt)

output_path = os.path.join(os.getcwd(),"Outputs",opt.folder_load)
mkdir(output_path)
save_images(dataset,output_path)

