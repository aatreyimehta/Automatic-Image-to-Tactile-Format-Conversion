#importing python packages 
import argparse
import os
import json
import re
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image, ImageFilter
from PIL.ImageOps import invert
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, ToPILImage
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
def load_model(model_path, opt,device):
    gen = create_gen(opt.gen,opt.input_dim,opt.output_dim,multigpu=False)
    gen.to(device)
    
    checkpoint = torch.load(model_path)
    gen.load_state_dict(checkpoint["gen"], strict=False)
    return gen


def load_data(photo_path,opt, mode='test', shuffle=False):
    data = get_dataset(photo_path, opt, mode=mode)
    dataset = DataLoader(dataset=data, batch_size=1, shuffle=shuffle, num_workers=4)
    return dataset

#loading numpy arrays from various dictionaries
def load_arrays(path):
    gen_loss = np.load(os.path.join(path, "genloss.npy"))
    disc_loss = np.load(os.path.join(path, "discloss.npy"))
    l1_loss = np.load(os.path.join(path, "l1loss.npy"))
    gp_loss = np.load(os.path.join(path, "gploss.npy"))
    per_loss = np.load(os.path.join(path, "perloss.npy"))
    return {"gen":gen_loss, "disc":disc_loss, "l1":l1_loss, "gp":gp_loss, "per": per_loss}


def visualize(out):
    ax_msk = invert(ToPILImage()(out[0]))   #binary mask indicating axis location in the output image
    grid_msk = ToPILImage()(out[1])   #binary mask indicating grid location in the output image
    content_msk = ToPILImage()(out[2])   #binary mask indicating content location in the output image
    
    #ax, content, grid are created from PIL images with extra dimension added
    ax = np.expand_dims(np.array(ax_msk), axis=2)
    content = np.expand_dims(np.array(content_msk), axis=2)
    grid = np.expand_dims(np.array(grid_msk), axis=2)

    #creating a blank array containing all zeros 
    blk = np.zeros((256,256,3), dtype=np.uint8)
    
    ax = np.concatenate((ax,ax,ax), axis=2)

    content = np.concatenate((content, blk), axis=2)
    grid = np.concatenate((blk, grid), axis=2)
    
    #ax, content, grid are converted back to PIL images
    ax = Image.fromarray(ax)
    content = Image.fromarray(content)
    grid = Image.fromarray(grid)
    
    ax.paste(grid, (0,0), grid_msk)
    ax.paste(content, (0,0), content_msk)
    
    return ax   #returning the resulting image


#concatenating horizontally or vertically based on mode parameter
def concat_images(*photos, mode="h"):
    #torch.cat((photo,sketch,output),2)
    if mode=="h":
        res = Image.new(photos[0].mode, (photos[0].width*len(photos),photos[0].height))
        for i in range(len(photos)):
            res.paste(photos[i], (photos[i].width*i,0))
    else:
        res = Image.new(photos[0].mode, (photos[0].width, photos[0].height*len(photos)))
        for i in range(len(photos)):
            res.paste(photos[i], (0, photos[i].height*i))

    return res

#saving a plot of the losses during training
def save_plot(loss_dict, opt):
	x = np.array(range(opt.epoch_count, opt.epoch_count+opt.total_iters))
	legends = loss_dict.keys()
	for y in loss_dict.values():
		plt.plot(x,y)
	plt.legend(legends)
	plt.xlabel("iteration")
	plt.ylabel("loss")
	plt.savefig(os.path.join(os.getcwd(),"models",opt.folder_load,"loss.png"))

#evaluating PyTorch model on a given dataset and computing the pixel accuracy, Dice coefficient, and Jaccard index
def eval_model(model, dataset, path):
    jaccard = []
    dice = []
    accuracy = []
    
    for i, batch in enumerate(dataset):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = model(real_A.to(device)).cpu()

        fake_axis = np.array(ToPILImage()(real_B[0][0]).convert('1'),dtype=np.uint8).flatten()
        #print(fake_axis)
        fake_grid = np.array(ToPILImage()(real_B[0][1]).convert('1'),dtype=np.uint8).flatten()
        #print(len(fake_grid))
        fake_cont = np.array(ToPILImage()(real_B[0][2]).convert('1'),dtype=np.uint8).flatten()

        gen_axis = np.array(ToPILImage()(out[0][0]).convert('1'), dtype=np.uint8).flatten()
        #print(gen_axis)
        gen_grid = np.array(ToPILImage()(out[0][0]).convert('1'), dtype=np.uint8).flatten()
        #print(len(gen_grid))
        gen_cont = np.array(ToPILImage()(out[0][0]).convert('1'), dtype=np.uint8).flatten()
        
        cm_axis = confusion_matrix(fake_axis, gen_axis)
        cm_grid = confusion_matrix(fake_grid, gen_grid)
        #print(cm_grid)
        cm_cont = confusion_matrix(fake_cont, gen_cont)
        
        
        j_axis = cm_axis[1,1]/(cm_axis[1,1] + cm_axis[0,1] + cm_axis[1,0])
        #if len(cm_grid)==2 and len(cm_grid[0])==2:
        j_grid = cm_grid[1,1]/(cm_grid[1,1] + cm_grid[0,1] + cm_grid[1,0])
        j_cont = cm_cont[1,1]/(cm_cont[1,1] + cm_cont[0,1] + cm_cont[1,0])
        jaccard.append((j_axis+j_grid+j_cont)/3)
        
        d_axis = cm_axis[1,1]/(cm_axis[1,1] + 0.5*(cm_axis[0,1] + cm_axis[1,0]))
        #if len(cm_grid)==2 and len(cm_grid[0])==2:
        d_grid = cm_grid[1,1]/(cm_grid[1,1] + 0.5*(cm_grid[0,1] + cm_grid[1,0]))
        d_cont = cm_cont[1,1]/(cm_cont[1,1] + 0.5*(cm_cont[0,1] + cm_cont[1,0]))
        dice.append((d_axis+d_grid+d_cont)/3)
        
        a_axis = (cm_axis[1,1]+cm_axis[0,0])/np.sum(cm_axis)
        #if len(cm_grid)==2 and len(cm_grid[0])==2:
        a_grid = (cm_grid[1,1]+cm_grid[0,0])/np.sum(cm_grid)
        a_cont = (cm_cont[1,1]+cm_cont[0,0])/np.sum(cm_cont)
        accuracy.append((a_axis+a_grid+a_cont)/3)
    
    a = f"Pixel Accuracy => min:{np.min(accuracy)}, max:{np.max(accuracy)}, avg:{np.mean(accuracy)}, std:{np.std(accuracy)}\n"
    d = f"Dice Coeff => min:{np.min(dice)}, max:{np.max(dice)}, avg:{np.mean(dice)}, std:{np.std(dice)}\n"
    j = f"Jaccard Index => min:{np.min(jaccard)}, max:{np.max(jaccard)}, avg:{np.mean(jaccard)}, std:{np.std(jaccard)}\n"
    with open(os.path.join(os.getcwd(),"models",opt.folder_load,"eval.txt"), 'w') as f:
        f.writelines([a,d,j])
    print (f"Acc: {np.mean(accuracy)}, IoU: {np.mean(jaccard)}, Dice: {np.mean(dice)}")

def save_images(model, dataset, path):
    for i, batch in enumerate(tqdm(dataset)):
        real_A, real_B = batch[0], batch[1]
        with torch.no_grad():
            out = model(real_A.to(device)).cpu()

        #a = unnormalize(real_A[0])
        a = real_A[0]   #source image
        b = real_B[0]   #target image
        out = out[0]   #output of trained data
        
        a_img = visualize(a)
        b_img = visualize(b)
        out_img = visualize(out)
        
        out_img.save(os.path.join(path,f"o_{i+1}.png"))
        empty_image = Image.new('RGB', (256, 256), color='white')   #an empty white image
        a_elements = concat_images(empty_image, ToPILImage()(a[0]), empty_image)   
        b_elements = concat_images(ToPILImage()(b[0]), ToPILImage()(b[1]), ToPILImage()(b[2]))
        out_elements = concat_images(ToPILImage()(out[0]), ToPILImage()(out[1]), ToPILImage()(out[2]))
        concat_images(a_elements, b_elements,out_elements, mode="v").save(os.path.join(path,f"elm_{i+1}.png"))   #vertically concatenating source, target and output together
        # print(f"file x_{i+1}.png saved.")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="pix2obj", help="The folder path including params.txt")  
    opt = parser.parse_args()

    opt_path = os.path.join(os.getcwd(),"models", opt.folder.split("/")[-1], "params.txt")
    opt = load_opt(opt_path)
    device = torch.device("cuda:0")

    model_path = os.path.join(os.getcwd(),"models",opt.folder_load,"final_model.pth")
    gen = load_model(model_path,opt,device)

    photo_path_test= os.path.join(os.getcwd(),opt.data,"test","source")
    dataset = load_data(photo_path_test,opt, shuffle=False)

    loss_path = os.path.join(os.getcwd(), "models", opt.folder_load)
    losses = load_arrays(loss_path)
    save_plot(losses, opt)

    output_path = os.path.join(os.getcwd(),"Outputs",opt.folder_save)
    mkdir(output_path)
    eval_model(gen, dataset,output_path)
    save_images(gen, dataset,output_path)

    # Set the pix2obj path as input directory
    input_directory = "/home/student/Project/Pix2Pix-obj (copy 1)/Outputs/pix2obj/"
    
    # Set the pix2obj directory path to save the output images
    output_directory = "/home/student/Project/Pix2Pix-obj (copy 1)/Outputs/pix2obj/"
    
    # Loop through all files in the input directory to convert all images in to grayscale
    for filename in os.listdir(input_directory):
    	if filename.endswith(".tiff") or filename.endswith(".png"):
    		# Open the image file
    		image = Image.open(os.path.join(input_directory, filename))
    		
    		# Convert the image to black and white
    		gray_image = image.convert('L')
    		
    		# Save the black and white image in the output directory
    		output_path = os.path.join(output_directory, filename)
    		gray_image.save(output_path)
