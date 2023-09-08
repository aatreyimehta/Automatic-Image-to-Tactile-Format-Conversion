#importing python packages to begin with
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from weakref import ref

from utils import draw_grids, postprocessing, maskgen
from polygon_gen import generate_polygon
from bezier_generator import generate_bezier
import matplotlib.font_manager

matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


def draw_pair(color, grid_param=0.4, figsize=(5,5), filename=None, **kwargs):
    grid_p = np.random.rand() 
    #Plotting source image for Bezier data
    with plt.style.context('default'):
    
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()   #get current axis object
        bg_color = "#"+''.join([random.choice('0123456789abcdef') for _ in range(6)])
        ax.set_facecolor(f'{bg_color}11')   #sets the face color of axis
        ax.spines['right'].set_visible(False)   #removes the right spine
        ax.spines['top'].set_visible(False)   #removes the top spine
        ax.set_axisbelow(True)
        ax.spines['left'].set_position('zero')   #sets the position of left spine to the zero coordinate
        ax.spines['left'].set_zorder(2)   #sets the zorder of left spine to 2
        ax.spines['left'].set_linewidth(random. randint(1, 4))   #width of y-axis

        ax.spines['bottom'].set_position('zero')    #sets the position of bottom spine to the zero coordinate
        ax.spines['bottom'].set_zorder(2)   #sets the zorder of bottom spine to 2
        ax.spines['bottom'].set_linewidth((random. randint(1, 4))) #width of x-axis

        ax.plot((1), (0), ls="", marker=random.choice(open("xaxis.txt","r").readline().split()), ms=10, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False, zorder=2)
        ax.plot((0), (1), ls="", marker=random.choice(open("yaxis.txt","r").readline().split()), ms=10, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False, zorder=2)
        ax.tick_params(length=5)   #length of ticks
       
             
        #n_rows = 2
        #n_col = 2
        #fig, axes = plt.subplots(n_rows, n_col)
        #for row_num in range(n_rows):
            #for col_num in range(n_col):
               #ax = axes[row_num][col_num]
               #ax.legend()  

	#adding bezier curves to the plot
        if "bezier" in kwargs:
            b = kwargs["bezier"]
            plt.plot(b[:,0], b[:,1], c=color, zorder=3)

        #adding scatter points to the plot
        if "scatter" in kwargs and kwargs["scatter"] is not None:
            points = kwargs["scatter"]
            plt.scatter(points[:,0], points[:,1], s=50, c=color, zorder=4, marker=random.choice(open("scatter.txt","r").readline().split()))

	#adding polygons to the plot
        if "polygon" in kwargs:
            polygon = kwargs["polygon"]
            plt.fill(polygon[:,0], polygon[:,1], fc=f"{color}33", ec=color, zorder=3)
       

        if grid_p < grid_param:
            draw_grids(ax, linestyle=random.choice(open("grid.txt","r").readline().split()))   #randomly selects the grid style from grid.txt for every individual plot
         
        plt.legend(['x-axis', 'y-axis', random.choice(open("legend.txt","r").readline().split()), 'inflection point'], loc ='best', prop={'family':random.choice(open("font.txt","r").readline().split())})   #layout for legend
        plt.title(random.choice(open("randomFile.txt","r").readline().split()), fontsize = random.randint(14,20), font =random.choice(open("font.txt","r").readline().split()))   #layout for chart title
        plt.xlabel(random.choice(open("label.txt","r").readline().split()), fontsize = random.randint(10,15), font =random.choice(open("font.txt","r").readline().split()))   #x label is randomly selected from label.txt file
        plt.ylabel(random.choice(open("label.txt","r").readline().split()), fontsize = random.randint(10,15), font =random.choice(open("font.txt","r").readline().split()))   #y label is randomly selected from label.txt file
        
        #saving the source images in source folder 
	fig.savefig(f'./source/s_{filename}.png', dpi=75)
        postprocessing(f'./source/s_{filename}.png')
        plt.close('all')


    #Plotting channelwise target image for Bezier data
    with plt.style.context('default'):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_xticklabels('')   #labels of x-ticks is set to an empty string
        ax.set_yticklabels('')   #labels of y-ticks is set to an empty string
        ax.spines['right'].set_visible(False)   #removing the right spine
        ax.spines['top'].set_visible(False)   #removing the top spine
        ax.set_axisbelow(True)
        ax.tick_params(direction='inout', length=20, width=1)

	#adding bezier curves to the plot
        if "bezier" in kwargs:
            b = kwargs["bezier"]
            plt.plot(b[:,0], b[:,1], c="w", lw=1, zorder=0)

	#adding scatter points to the plot
        if "scatter" in kwargs and kwargs["scatter"] is not None:
            points = kwargs["scatter"]
            plt.scatter(points[:,0], points[:,1], s=1, c="w", zorder=0)

	#adding polygons to the plot
        if "polygon" in kwargs:
            polygon = kwargs["polygon"]
            plt.fill(polygon[:,0], polygon[:,1], fc="#ffffff00", ec="w", zorder=0)

        #setting the axis spine positions
        ax.spines['left'].set_position('zero')
        ax.spines['left'].set_zorder(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_position('zero')
        ax.spines['bottom'].set_zorder(2)
        ax.spines['bottom'].set_linewidth(2)
        
        #plt.xlabel(random.choice(open("myFile.txt","r").readline().split()))
        #plt.ylabel(random.choice(open("myFile.txt","r").readline().split()))


	#setting further axis properties
        ax.plot((1), (0), ls="", ms=10, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False, zorder=2)
        wr1 = ref(ax.lines[-1])
        ax.plot((0), (1), ls="", ms=10, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False, zorder=2)
        wr2 = ref(ax.lines[-1])
        fig.savefig(f'./tactile/t_{filename}_axes.tiff', dpi=75)   #saving the axes part as an image
        
        ax.tick_params(direction='inout', length=0, width=0, zorder=3)
        ax.lines.remove(wr1())   
        ax.lines.remove(wr2())
        for _, item in ax.spines.items():
            item.set_visible(False)

	#setting grid properties if grids are produced
        if grid_p < grid_param:
            draw_grids(ax, color='k', linestyle='--', linewidth=1)
            
        
        #saving the grids part as an image
        fig.savefig(f'./tactile/t_{filename}_grids.tiff', dpi=75)
        plt.grid(False)


	#setting bezier data properties
        if "bezier" in kwargs:
            plt.plot(b[:,0], b[:,1], clip_on=False, c='k', lw=4, zorder=3)
        if "scatter" in kwargs and kwargs["scatter"] is not None:
            plt.scatter(points[:,0], points[:,1], s=300, c='k', ec='w', lw=5, zorder=4)

        if "polygon" in kwargs:
            polygon = kwargs["polygon"]
            plt.fill(polygon[:,0], polygon[:,1], fc="#ffffff00", ec='k', lw=4, zorder=3)


	#saving the content of bezier data as an image
        fig.savefig(f'./tactile/t_{filename}_content.tiff', dpi=75)
        maskgen(f'./tactile/t_{filename}.tiff')
        plt.close('all')

  #Plotting grayscale/rgb target for Bezier data
    with plt.style.context('default'):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_xticklabels('')   #labels of x-axis is set to an empty string
        #ax.xticklabels('').set_visible(True)
        ax.set_yticklabels('')   #labels of x-axis is set to an empty string
        ax.spines['right'].set_visible(False)   #right spine is removed
        ax.spines['right'].set_color('white')   #color of right spine is set to white
        ax.spines['top'].set_visible(False)   #top spine is removed
        ax.spines['top'].set_color('white')   #color of top spine is set to white
        #ax.spines['bottom'].set_visible(True)
        #ax.spines['left'].set_visible(True)
        ax.set_axisbelow(True)
        ax.tick_params(direction='inout', length=10, width=1)
        print(ax.tick_params)
        #ax.tick_params(direction='inout', length=0, width=1)
        #ax.set_xticks([5,10,15])
        

	#adding the bezier curves in the plot
        if "bezier" in kwargs:
            b = kwargs["bezier"]
            plt.plot(b[:,0], b[:,1], c="w", lw=1, zorder=0)

	#adding the scatter points in the plot
        if "scatter" in kwargs and kwargs["scatter"] is not None:
            points = kwargs["scatter"]
            plt.scatter(points[:,0], points[:,1], s=1, c="w", zorder=0)

	#adding polygons in the plot
        if "polygon" in kwargs:
            polygon = kwargs["polygon"]
            plt.fill(polygon[:,0], polygon[:,1], fc="#ffffff00", ec="w", zorder=0)

	#setting the spine positions
        ax.spines['left'].set_position('zero')
        ax.spines['left'].set_zorder(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_position('zero')
        ax.spines['bottom'].set_zorder(2)
        ax.spines['bottom'].set_linewidth(2)
        
        #plt.xlabel(random.choice(open("myFile.txt","r").readline().split()))
        #plt.ylabel(random.choice(open("myFile.txt","r").readline().split()))

        ax.plot((1), (0), ls="", ms=10, color="k",
        transform=ax.get_yaxis_transform(), clip_on=False, zorder=2)
        wr1 = ref(ax.lines[-1])
        ax.plot((0), (1), ls="", ms=10, color="k",
        transform=ax.get_xaxis_transform(), clip_on=False, zorder=2)
        wr2 = ref(ax.lines[-1])
        plt.axes()
        #ax.tick_params(direction='inout', length=0, width=0, zorder=3 )
        ax.lines.remove(wr1())
        ax.lines.remove(wr2())
        for _, item in ax.spines.items():
            item.set_visible(True)   #produces axis for the plots
            
            
        #fig_tactile_size = random.choices([[2.9,2.9], [2.9,2.9], [2.9,2.9]], weights=opt.p_figsize)[0] 

        if grid_p < grid_param:
            draw_grids(ax, color='k', linestyle='--', linewidth=1)
            plt.grid(True)   #produces grids for the plots

	#setting the bezier content data for the plots
        if "bezier" in kwargs:
            plt.plot(b[:,0], b[:,1], clip_on=False, c='k', lw=4, zorder=3)
        if "scatter" in kwargs and kwargs["scatter"] is not None:
            plt.scatter(points[:,0], points[:,1], s=300, c='k', ec='w', lw=5, zorder=4)
        if "polygon" in kwargs:
            polygon = kwargs["polygon"]
            plt.fill(polygon[:,0], polygon[:,1], fc="#ffffff00", ec='k', lw=4, zorder=3)
            
        #plt.rcParams['figure.figsize'] = [2, 2]
        
        #fig = matplotlib.pyplot.gcf()
        #fig.set_size_inches(3.4133333333333336, 3.4133333333333336)

        #fig.savefig(f'./tactile/t_{filename}.tiff', dpi=96)
        #fig.savefig(f'./tactile/t_{filename}.tiff', dpi=75)
        fig.savefig(f'./tactile1/t_{filename}.tiff', dpi=75)   #saves the whole axis, content and grid accumulated as one image in tactile1 folder
        
        #maskgen(f'./tactile1/t_{filename}.tiff')
        plt.close('all')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnt_bezier", type=int, default=2000, help="number of bezier curves")
    parser.add_argument("--cnt_scatter", type=int, default=1500, help="number of scatter points")
    parser.add_argument("--cnt_polygon", type=int, default=1500, help="number of polygons")
    parser.add_argument("--p_figsize", nargs=3, type=float, default=[.5, .25, .25], help="figure size probabilities")
    parser.add_argument("--p_1D", type=float, default=0.4, help="probability of 1D bezeir generation")
    parser.add_argument("--p_grid", type=float, default=0.4, help="probability of drawing grid")

    opt = parser.parse_args()

    os.makedirs('./source', exist_ok=True)   #creates a source folder
    os.makedirs('./tactile', exist_ok=True)   #creates a tactile folder
    os.makedirs('./tactile1', exist_ok=True)   #creates a tactile1 folder


    lim_bezier = opt.cnt_bezier   #max number of curves that will be generated will be equal to the value of opt.cnt_bezier
    lim_polygon = opt.cnt_bezier + opt.cnt_polygon   #max number of polygons that will be generated will be equal to the value of opt.cnt_bezier and opt.cnt_polygon
    lim_scatter = opt.cnt_bezier + opt.cnt_polygon + opt.cnt_scatter   #max number of scatter points that will be generated will be equal to the value of opt.cnt_polygon and opt.cnt_scatter

    for i in tqdm(range(lim_bezier), desc="bezier curves"):
        clr = "#"+''.join([random.choice('0123456789abcdef') for _ in range(6)])
        fig_size = random.choices([[5,5], [2.5,5], [5,2.5]], weights=opt.p_figsize)[0]   #figure ratio is randomly selected from 1:1, 1:2 and 2:1 options
        
        # uncomment to generate fresh bezier data
        x = np.linspace(0, 1, 10000).reshape(-1,1)
        p = np.array([[random.randint(-20,20), random.randint(-20,20)] for i in range(random.randint(2,20))])
        if random.random() <= opt.p_1D:
            p = p.reshape(-1,1).flatten()[::2]
        b = generate_bezier(x, p)
        print('b', len(b))
        #b = np.load(f"./points/{i+1}.npy")
        pointidx = np.random.randint(10)
        ps = b[0::b.shape[0]//pointidx,:] if pointidx > 0 else None
        draw_pair(clr,opt.p_grid,fig_size, f"{i+1}", bezier=b, scatter=ps)

   #generating plots with different elements like curves, scatter points and polygons with random parameters     
    for i in tqdm(range(lim_bezier, lim_polygon), desc="polygons"):
        clr = "#"+''.join([random.choice('0123456789abcdef') for _ in range(6)])
        fig_size = random.choices([[5,5], [2.5,5], [5,2.5]], weights=opt.p_figsize)[0]

        ps = generate_polygon(center=(random.random()*2-1, random.random()*2-1),
                            avg_radius=1.5,
                            irregularity=0.2,
                            spikiness=0.1,
                            num_vertices=np.random.randint(3,10))
        draw_pair(clr,opt.p_grid,fig_size, f"{i+1}", scatter=ps, polygon=ps)


    for i in tqdm(range(lim_polygon, lim_scatter), desc="scatter plots"):
        clr = "#"+''.join([random.choice('0123456789abcdef') for _ in range(6)])
        fig_size = random.choices([[5,5], [2.5,5], [5,2.5]], weights=opt.p_figsize)[0]
        idx = np.random.randint(2,20)
        ps = np.array([[random.random()*100-50, random.random()*100-50] for _ in range(idx)])
        draw_pair(clr,opt.p_grid,fig_size, f"{i+1}", scatter=ps)
