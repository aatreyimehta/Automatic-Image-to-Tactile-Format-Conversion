# !pip install retrying
# !pip install kaleido
# !pip install plotly==5.8

#importing python packages to begin with
import os
import shutil
import random
import numpy as np
import pandas as pd
import retrying
import random
from random import randint
import random as rand
import tensorflow as tf
import string
from tqdm import tqdm
import plotly.express as px

import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.io._orca

tf.get_logger().setLevel('ERROR')

# Launch orca
unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
plotly.io._orca.request_image_with_retrying = wrapped

MAX_Y = 100
MIN_Y = 0


# These set the ordering of bars: Ascending, descending or normal
PLOT_ORDER = {"0": 0, "1": 0, "2": 0}
SIZEREFS = {"1": 7.0, "2": 2.5 , "3": 2.0}

#generating random data used to plot bar charts
def generate_metadata(min_y = 0, max_y = 100):
    num_bars = 5
    x = list(range(1, num_bars*2, 2))   #generate x values only for odd ticks to maintain bar gap for tall but narrow width barcharts
    
    num_groups = np.random.randint(low=1, high=1+1)
    
    # Generate random values for bars
    y = np.random.randint(low = min_y, high = max_y + 1, size=(num_groups, num_bars))

    # Sort y plots accordingly
    sort_plots(y) 
    
    return {'x': x, 'y': y}
    
#sorting the arrays in either ascending or descending order based on value of PLOT_ORDER
def sort_plots(plots):
    for i in range(len(plots)):
        t = plots[i]
        if PLOT_ORDER[str(i)] % 3 == 1:
            t = np.sort(t)
        elif PLOT_ORDER[str(i)] % 3 == 2:
            t = np.sort(t)
            t = np.flip(t)
            
        plots[i] = t
        PLOT_ORDER[str(i)] += 1
 
#generating data for various samples of bar charts
def generate_data(num_samples = 1):
    data = list()
    metadata = list()
    circle_data = list()
    
    for i in tqdm(range(num_samples), desc="generating metadata: "):
        
        sample = {}
        
        sample_metadata = generate_metadata()
        
        
        sample["y_values"] = sample_metadata["y"]
        sample["x_values"] = sample_metadata["x"]
              
        num_bars = len(sample["y_values"][0])
        num_groups = len(sample["y_values"])
        min_x = np.amin(sample["x_values"])
        max_x = np.amax(sample["x_values"])
        min_y = np.amin(sample["y_values"])
        max_y = np.amax(sample["y_values"])
        
        sample_styles = generate_styles(num_bars=num_bars, num_groups = num_groups, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        
        sample["marker_colors"] = sample_styles["marker_colors"]
        sample["marker_colors_rgba"] = sample_styles["marker_colors_rgba"]
        sample["plot_bg_color"] = sample_styles["plot_bg_color"]
        sample["plot_bg_color_rgba"] = sample_styles["plot_bg_color_rgba"]
        sample["bargap"] = sample_styles["bargap"]
        sample["bargroupgap"] = sample_styles["bargroupgap"]
        sample["x_tick_start"] = sample_styles["xaxis"]["tick0"]
        sample["x_tick_dist"] = sample_styles["xaxis"]["dtick"]
        sample["y_tick_start"] = sample_styles["yaxis"]["tick0"]
        sample["y_tick_dist"] = sample_styles["yaxis"]["dtick"]
        
        circle_data_sample = generate_circle_bars(sample)
        
        data.append(sample)
        metadata.append(sample_metadata)
        circle_data.append(circle_data_sample)
        
    return data, metadata, circle_data

#generating marker colors for bars in a bar chart
def generate_marker_colors(num_groups, num_bars):
    one_color = np.random.randint(2) # Generate one color or different colors for bars
    
    # Avoid full black or full white
    if num_groups > 1:   #if num_groups > 1, it generates num_groups of distinct colors
        marker_colors = [(np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.round(np.random.uniform(0.5, 1), 2)) for i in range(num_groups)]
        marker_colors_as_str = ["rgba" + str(c) for c in marker_colors]  
    elif one_color:
        marker_colors = (np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.round(np.random.uniform(0.5, 1), 2))
        marker_colors_as_str = "rgba" + str(marker_colors)
    else:
        marker_colors = [(np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.random.randint(low=1, high=255), np.round(np.random.uniform(0.5, 1), 2)) for i in range(num_bars)]
        marker_colors_as_str = ["rgba" + str(c) for c in marker_colors]
        
    return marker_colors, marker_colors_as_str

#computing the reference size for markers in a bar chart depending on the size of the plot and the range of the y-axis values
def compute_sizeref(plot_size = 500, num_groups = 1, rangey = 100):
    width = (plot_size / rangey) / (num_groups * 2 - 1)
    return ((2.0 *  width)/((3 * width)**2))

#computing the width of scatter markers(circles) to be used for the scatter plots
def compute_scatter_width(plot_size = 500, num_groups = 1, num_bars = 20):
    div = 4 if num_groups == 2 else 3
    temp_width = (plot_size / (((num_bars - 1) * (num_groups + (num_groups / div))) + num_groups)) / num_groups
    width = 7.5 if temp_width > 7.5 else temp_width
    if width >= 7.5 and num_groups == 2:
        width = 5.5
    return SIZEREFS[str(num_groups)] * (20.0 / num_bars)

#generating random bar widths for given number of bars
def generate_bar_widths(num_bars):
    one_width = np.random.randint(2) # Generate one width or different widths for bars
    if one_width:
        widths = np.random.uniform(0.1, 1)
    else:
        widths = np.random.uniform(low = 0.1, high = 1, size=num_bars)
    return widths

#generating appropriate bar gaps
def generate_bar_gap():
    return np.round(np.random.uniform(0, 0.3), 2)

#generating appropriate gaps between bar groups
def generate_bar_group_gap():
    return np.round(np.random.uniform(0, 0.3), 2)

#generating starting tick and size for both the x and y axis in a plot
def generate_tick_size(min, max):
    if min <= 0:
        start = 0
    else: 
        start = np.random.randint(0, min)
    
    # Minimum number of ticks is 2, max is 20 
    size = np.round((max - min) / np.random.randint(2, 20 + 1), 2)
    return start, size

#generating background color for the plot
def generate_plot_bgcolor():
    is_white = np.random.randint(2) # White or any color
    if is_white:
        bg_color = (255, 255, 255, 1)
    else:
        bg_color = (np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.round(np.random.uniform(0, 0.35), 2))
        

    bg_color_rgba = "rgba" + str(bg_color)
    
    return bg_color, bg_color_rgba

#generating paper color for the plot
def generate_plot_paper_color():
    is_white = np.random.randint(2) # White or any color
    if is_white:
        paper_color = (255, 255, 255, 1)
    else:
        paper_color = (np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.random.randint(low=0, high=255 + 1), np.round(np.random.uniform(0, 0.35), 2))
        
    fig, ax = plt.subplots(figsize=(512, 512))
    fig.set_aspect('equal')

    paper_color_rgba = "rgba" + str(paper_color)
    
    return paper_color, paper_color_rgba

#generating points for circle bar chart with a single color
def generate_circle_bars_one(data):
    num_bars = len(data["x_values"])
    num_groups = len(data["y_values"])
    width  = 30.75 # compute_scatter_width(num_groups = num_groups, num_bars = num_bars)
    group_points = []
    
    for i in range(num_groups):
        bar_points = []
        for j in range(num_bars):
            x = data["x_values"][j] + (data["x_values"][j] * 0.0)
            pointsx = [x]
            pointsy = [0]
            widths = [0]
            endy = data["y_values"][i][j]
            curryst = 4.25
            
            if endy < 2.5:
                continue

            while(curryst <= endy):
                pointsx.append(x)
                pointsy.append(curryst)
                widths.append(width)

                curryst = curryst + 7.5

            bar_points.append({"x": pointsx, "y": pointsy, "widths": widths})

        group_points.append(bar_points)
        
    return group_points

#generating points for circle bar chart with two colors
def generate_circle_bars_two(data):
    num_bars = len(data["x_values"])
    num_groups = len(data["y_values"])
    width  = 11.5 # compute_scatter_width(num_groups = num_groups, num_bars = num_bars)
    group_points = []
    
    for i in range(num_groups):
        bar_points = []
        for j in range(num_bars):
            x = data["x_values"][j] + (i * 0.35)
            pointsx = [x]
            pointsy = [0]
            widths = [0]
            endy = data["y_values"][i][j]
            curryst = 2

            while(curryst < endy):
                pointsx.append(x)
                pointsy.append(curryst)
                widths.append(width)

                curryst = curryst + 3

            bar_points.append({"x": pointsx, "y": pointsy, "widths": widths})

        group_points.append(bar_points)
        
    return group_points
"""
def generate_circle_bars_three(data):
    num_bars = len(data["x_values"])
    num_groups = len(data["y_values"])
    width  = 8.125 #compute_scatter_width(num_groups = num_groups, num_bars = num_bars) 
    group_points = []
    
    for i in range(num_groups):
        bar_points = []
        for j in range(num_bars):
            x = data["x_values"][j] + (i * 0.25)
            pointsx = [x]
            pointsy = [0]
            widths = [0]
            endy = data["y_values"][i][j]
            curryst = 1.5

            while(curryst < endy):
                pointsx.append(x)
                pointsy.append(curryst)
                widths.append(width)

                curryst = curryst + 2.125

            bar_points.append({"x": pointsx, "y": pointsy, "widths": widths})

        group_points.append(bar_points)
        
    return group_points
"""
def generate_circle_bars(data):
    num_groups = len(data["y_values"])
    
    if num_groups == 1:
        return generate_circle_bars_one(data)
    if num_groups == 2:
        return generate_circle_bars_two(data)
    #if num_groups == 3:
        #return generate_circle_bars_three(data)

#generating styles like the color of the plot, the tick sizes for the axis and the bar gaps
def generate_styles(num_bars, num_groups, min_x, max_x, min_y, max_y):

    (marker_colors, marker_colors_rgba) = generate_marker_colors(num_groups, num_bars)
    (plot_bg_color, plot_bg_color_rgba) = generate_plot_bgcolor()
    bargap = generate_bar_gap()
    bargroupgap = generate_bar_group_gap()
    (xtick_start, xtick_size) = generate_tick_size(min_x, max_x)
    (ytick_start, ytick_size) = generate_tick_size(min_y, max_y)
    
    styles = {
        "marker_colors": marker_colors,
        "marker_colors_rgba": marker_colors_rgba,
        "plot_bg_color": plot_bg_color,
        "plot_bg_color_rgba": plot_bg_color_rgba,
        "bargap": bargap,
        "bargroupgap": bargroupgap,
        "xaxis": {
            "tick0": xtick_start,
            "dtick": xtick_size
        },
        "yaxis": {
            "tick0": ytick_start,
            "dtick": ytick_size
        }
    }
    
    return styles
  
    

#grid_dash=random.choice(open("grid.txt","r").readline().split())

num_ticks = random.randint(7,12)   #generating random number of ticks between numbers 7 and 12
    
#Plotting the source domain(horizontal barcharts) for 1024x512 image dimension i.e. 2:1 ratio
def write_source_data_2(data, filepath, figsize=(1024, 512), draw_grid=False, tick_step=10):

    linewidth = random.uniform(1,5)   #generating axis linewidth of random thickness for every image
    
    #initializing word, that selects random string of alphabets every time
    alphabet = string.ascii_lowercase
    word = ''.join(random.choice(alphabet) for i in range(randint(2,3)))  
    
    fig = go.Figure()

    #setting the x and y values for horizontal barchart
    for r in range(len(data["y_values"])):
        fig.add_trace(go.Bar(y=data["x_values"],
                        x=data["y_values"][r],
                        orientation="h",   #horizontal orientation
                        width=0.7,
                        #orientation=random.choice(open("orientation.txt","r").readline().split()),
                        name=word,
                        #name=random.choice(open("legend.txt","r").readline().split()),
                        marker_color=data["marker_colors_rgba"][r] if len(data["y_values"]) > 1 else data["marker_colors_rgba"],
                        marker_line_width=2
                        ))
                                         
    fig.update_layout(
        margin=dict(l=5, r=25, t=30, b=25),
        plot_bgcolor=data["plot_bg_color_rgba"],
        width=figsize[0], height=figsize[1],
        #title=random.choice(open("randomFile.txt","r").readline().split()),
        xaxis_range=[0,110],   #range for x axis
        yaxis_range=[-0.5,10],   #range for y axis
        xaxis_title=random.choice(open("label.txt","r").readline().split()),   #randomly selecting a word from file label.txt for xaxis title
        yaxis_title=random.choice(open("label.txt","r").readline().split()),   #randomly selecting a word from file label.txt for yaxis title
        #bargap=data["bargap"],
        bargroupgap=data["bargroupgap"],
        showlegend=True,
        #axis_width=randint(1,5),
        #setting properties for x axis
        xaxis={
                "showline": True, 
                #"linewidth": random.uniform(1,5), 
                "linewidth": linewidth,
                "linecolor": 'black',
                "dtick": tick_step
            },
        #setting properties for y axis    
        yaxis={
                "showline": True, 
                #"linewidth": random.uniform(1,5), 
                "linewidth": linewidth,
                "linecolor": 'black',
                "ticks": "outside",
                #"dtick": tick_step
            })

    fig.update_layout(bargap=0.11)
    #fig.add_annotation(ax=-2.5, axref='x', ay=0, ayref='y', x=9.8, xref='x', y=0, yref='y', arrowwidth=random.randint(2,5), arrowhead=2)
    #fig.add_annotation(ax=15, axref='x', ay=0, ayref='y', x=0, xref='x', y=0, yref='y', arrowwidth=random.randint(2,5), arrowhead=2)
                   	
    fig.update_layout(legend=dict(
    	orientation=random.choice(open("orientation.txt","r").readline().split()),
    	
    	font_family=random.choice(open("font.txt","r").readline().split()),
    	yanchor=random.choice(open("yanchor.txt","r").readline().split()),
    	xanchor=random.choice(open("xanchor.txt","r").readline().split())
    ))
    
    #layout for legend
    fig.update_layout(
    	yaxis = dict(
    	tickmode = 'array',
    	#tickvals = [20,40,60,80,100,120,140,160,180,200],
    	tickvals = [0.1,1,2,3,4,5,6,7,8,9],
        ticktext =[random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split()), random.choice(open("legend.txt","r").readline().split())]
    ))   #selecting random word every time from legend.txt file for the tick text
    
    #df = px.data.tips()     
    #fig = px.bar(
    	#df,
    	#orientation='h',
    	#)        
    
    
    fig.update_layout(
    	font=dict(
    		size=random.randint(13,22),   #selecting random size for fonts for the plots
    		family=random.choice(open("font.txt","r").readline().split())   #selecting random font family for the plot from font.txt
    		)
    	)
    
    #layout for the chart title
    fig.update_layout(
    	title_text=random.choice(open("randomFile.txt","r").readline().split()),   #chart title is selected randomly from randomFile.txt 
    	title_x=random.uniform(0.0,0.9),
    	title_font_size=random.randint(25,35),   #size of title font
    	margin={'t':50, 'b':50, 'l':1},   #adjusted so that the whole title is visible
    	title_font_family=random.choice(open("font.txt","r").readline().split())
    		) 
    
    		    	
        
    #plt.title(random.choice(open("randomFile.txt","r").readline().split()), fontsize = random.randint(11,18))
    #plt.xlabel(random.choice(open("myFile.txt","r").readline().split()), fontsize = random.randint(10,15))
    #plt.ylabel(random.choice(open("myFile.txt","r").readline().split()), fontsize = random.randint(10,15))  


    #properties to be included if grids are produced
    fig.update_xaxes(showgrid=False)
    if draw_grid:
        fig.update_xaxes(showgrid=True, gridcolor='#aaaaaa', gridwidth=1, griddash=random.choice(open("grid.txt","r").readline().split()))
        #fig.update_xaxes(showgrid=True, gridcolor='#aaaaaa', gridwidth=1, griddash= )
    #elif plot_bgcolor == 'white':
    	#fig.update_yaxes(showgris=True, gridcolor='#000000', gridwidth=1)

    #saving the plot as an image file
    pio.write_image(fig=fig, file=filepath, format="png", width=figsize[0], height=figsize[1])
    #plt.show()

#Plotting channelwise target images for 1024x512 image dimension  
def write_circle_target_data_2(data, filepath, figsize=(1024, 512), draw_grid=False, tick_step=10):
    fig = go.Figure()
    for i in range(len(data)):
        for j in range(len(data[i])):
            fig.add_trace(go.Scatter(x=data[i][j]["y"],
                        y=data[i][j]["x"],
                        marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[0]/figsize[1]),        
            "sizemode":'diameter',
            "sizeref": 0.71,   #size reference for 1024x512 images
                                 },
                        ))

    fig.update_traces(mode='markers', marker_line_width=1, visible=False,)


    fp_parts = filepath.rsplit(".", 1)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=30, b=30),
        width=figsize[0], height=figsize[1],
        xaxis_range=[0,110],   #range of x axis
        yaxis_range=[-0.5,10],   #range of y axis
        #orientation="h",   #horizontal orientation
        #barmode="relative",
        #setting x and y axis parameters 
        yaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black'},
        xaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black', "ticks": "outside", "dtick": tick_step, "ticklen": 10, "tickwidth": 2},
        showlegend=False
    )   
    
    #saving the axes part as an image
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_axes.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])

    #defining grid properties for when grids are produced
    if draw_grid:
        #fig.update_yaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=1)
        fig.update_xaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=1)
        
    fig.update_layout(yaxis={"linecolor": 'white'}, xaxis={"linecolor": 'white', "ticklen": 0, "tickwidth": 0})
    fig.update_layout(yaxis={"showgrid":True})
    #saving the grid part as an image
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_grids.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])

    fig.update_traces(mode='markers', marker_line_width=2, marker_color="white", marker_line_color="black", visible=True)
    #fig.update_layout(yaxis={"showgrid":False})
    fig.update_layout(xaxis={"showgrid":False})
    #saving the bar content an an image
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_content.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])
    
    
#Plotting grayscale/rgb target images for 1024x512 image dimension   
def write_circle_target_data1_2(data, filepath, figsize=(1024, 512), draw_grid=False, tick_step=10):
    fig = go.Figure()
    for i in range(len(data)):
        for j in range(len(data[i])):
            fig.add_trace(go.Scatter(x=data[i][j]["y"],
                        y=data[i][j]["x"],
                        marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[0]/figsize[1]),    
                        #marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[1]/figsize[0]),     
            "sizemode":'diameter',
            "sizeref": 0.71,   #size reference for 1024x512 image dimension
                                 },
                        ))
    
    fig.update_traces(mode='markers', marker_line_width=1, visible=False)

    fp_parts = filepath.rsplit(".", 1)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=30, b=30),
        width=figsize[0], height=figsize[1],
        xaxis_range=[0,110],   #setting same x axis range as source for the target images
        yaxis_range=[-0.5,10],   #setting same y axis range as source for the target images
        #setting axis properties for x and y axis
        yaxis={"showticklabels": False, "linewidth": 4, "linecolor": 'black'},
        xaxis={"showticklabels": False, "linewidth": 4, "linecolor": 'black', "ticks": "outside", "dtick": tick_step, "ticklen": 0, "tickwidth": 0, "ticklen": 10, "tickwidth": 4, "showgrid":False},
        showlegend=False
    )    
    fig.update_traces(mode='markers', marker_line_width=3.5, marker_color="white", marker_line_color="black", visible=True)
    if draw_grid:
    	fig.update_xaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=3)
    #saving the rgb/grayscale output as an image
    pio.write_image(fig=fig, file=f"{fp_parts[0]}.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])


#Plotting source domain for 760x512 image dimension(i.e. 1.5:1 ratio) for horizontal barcharts(all the source properties remain the same as 1024x512 image dimension, except for size reference for target images)
def write_source_data_3(data, filepath, figsize=(760, 512), draw_grid=False, tick_step=10):

    linewidth = random.uniform(1,5)
    
    fig = go.Figure()

    for r in range(len(data["y_values"])):
        fig.add_trace(go.Bar(y=data["x_values"],
                        x=data["y_values"][r],
                        orientation="h",
                        width=0.7,
                        #orientation=random.choice(open("orientation.txt","r").readline().split()),
                        name=random.choice(open("legend.txt","r").readline().split()),
                        marker_color=data["marker_colors_rgba"][r] if len(data["y_values"]) > 1 else data["marker_colors_rgba"],
                        marker_line_width=2
                        ))
                                         
    fig.update_layout(
        margin=dict(l=5, r=25, t=30, b=25),
        plot_bgcolor=data["plot_bg_color_rgba"],
        width=figsize[0], height=figsize[1],
        #title=random.choice(open("randomFile.txt","r").readline().split()),
        xaxis_range=[0,110],
        yaxis_range=[-0.5,10],
        xaxis_title=random.choice(open("label.txt","r").readline().split()),
        yaxis_title=random.choice(open("label.txt","r").readline().split()),
        #bargap=data["bargap"],
        bargroupgap=data["bargroupgap"],
        showlegend=True,
        #axis_width=randint(1,5),
        xaxis={
                "showline": True, 
                #"linewidth": random.uniform(1,5), 
                "linewidth": linewidth,
                "linecolor": 'black',
                "dtick": tick_step
            },
            
        yaxis={
                "showline": True, 
                #"linewidth": random.uniform(1,5), 
                "linewidth": linewidth,
                "linecolor": 'black',
                "ticks": "outside",
                #"dtick": tick_step
            })
            

    fig.update_layout(bargap=0.11)
    #fig.add_annotation(ax=-2.5, axref='x', ay=0, ayref='y', x=9.8, xref='x', y=0, yref='y', arrowwidth=random.randint(2,5), arrowhead=2)
    #fig.add_annotation(ax=15, axref='x', ay=0, ayref='y', x=0, xref='x', y=0, yref='y', arrowwidth=random.randint(2,5), arrowhead=2)
    
    with open("x.txt","r") as f:
    	x_options = f.readline().split()
    
    with open("y.txt","r") as f:
    	y_options = f.readline().split()
                   	
    fig.update_layout(legend=dict(
    	orientation=random.choice(open("orientation.txt","r").readline().split()),
    	font_family=random.choice(open("font.txt","r").readline().split()),
    	x=float(random.choice(x_options)),
    	y=float(random.choice(y_options))
    	#yanchor=random.choice(open("yanchor.txt","r").readline().split()),
    	#xanchor=random.choice(open("xanchor.txt","r").readline().split())
    ))

    alphabet = string.ascii_lowercase


    ticktext = []
    for i in range(10):
    	word = ''.join(random.choice(alphabet) for i in range(3))
    	ticktext.append(word)
        
    fig.update_layout(
    	yaxis = dict(
    	tickmode = 'array',
    	#tickvals = [20,40,60,80,100,120,140,160,180,200],
    	tickvals = [0.1,1,2,3,4,5,6,7,8,9],
        ticktext = ticktext
    ))  
    
    #df = px.data.tips()     
    #fig = px.bar(
    	#df,
    	#orientation='h',
    	#)        
    
    
    fig.update_layout(
    	font=dict(
    		size=random.randint(13,22),
    		family=random.choice(open("font.txt","r").readline().split())
    		)
    	)
    	
    fig.update_layout(
    	title_text=random.choice(open("randomFile.txt","r").readline().split()),
    	title_x=random.uniform(0.0,0.9),
    	title_font_size=random.randint(25,35),
    	margin={'t':50, 'b':50, 'l':1}, #so that the whole title is visible
    	title_font_family=random.choice(open("font.txt","r").readline().split())
    		) 
    
    		    	
        
    #plt.title(random.choice(open("randomFile.txt","r").readline().split()), fontsize = random.randint(11,18))
    #plt.xlabel(random.choice(open("myFile.txt","r").readline().split()), fontsize = random.randint(10,15))
    #plt.ylabel(random.choice(open("myFile.txt","r").readline().split()), fontsize = random.randint(10,15))  


    fig.update_xaxes(showgrid=False)
    if draw_grid:
        fig.update_xaxes(showgrid=True, gridcolor='#aaaaaa', gridwidth=1, griddash=random.choice(open("grid.txt","r").readline().split()))
        #fig.update_xaxes(showgrid=True, gridcolor='#aaaaaa', gridwidth=1, griddash= )
    #elif plot_bgcolor == 'white':
    	#fig.update_yaxes(showgris=True, gridcolor='#000000', gridwidth=1)

    pio.write_image(fig=fig, file=filepath, format="png", width=figsize[0], height=figsize[1])
    #plt.show()

#Plotting channelwise target image for 760x512 dimension
def write_circle_target_data_3(data, filepath, figsize=(760, 512), draw_grid=False, tick_step=10):
    fig = go.Figure()
    for i in range(len(data)):
        for j in range(len(data[i])):
            fig.add_trace(go.Scatter(x=data[i][j]["y"],
                        y=data[i][j]["x"],
                        marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[0]/figsize[1]),        
            "sizemode":'diameter',
            "sizeref": 0.87,   #size reference for 760x512 dimension
                                 },
                        ))

    fig.update_traces(mode='markers', marker_line_width=1, visible=False,)


    fp_parts = filepath.rsplit(".", 1)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=30, b=30),
        width=figsize[0], height=figsize[1],
        xaxis_range=[0,110],
        yaxis_range=[-0.5,10],
        #orientation="h",
        #barmode="relative",
        yaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black'},
        xaxis={"showticklabels": False, "linewidth": 2, "linecolor": 'black', "ticks": "outside", "dtick": tick_step, "ticklen": 10, "tickwidth": 2},
        showlegend=False
    )   
    
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_axes.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])

    if draw_grid:
        #fig.update_yaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=1)
        fig.update_xaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=1)
        
    fig.update_layout(yaxis={"linecolor": 'white'}, xaxis={"linecolor": 'white', "ticklen": 0, "tickwidth": 0})
    fig.update_layout(yaxis={"showgrid":True})
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_grids.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])

    fig.update_traces(mode='markers', marker_line_width=2, marker_color="white", marker_line_color="black", visible=True)
    #fig.update_layout(yaxis={"showgrid":False})
    fig.update_layout(xaxis={"showgrid":False})
    pio.write_image(fig=fig, file=f"{fp_parts[0]}_content.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])
    
    
#Plotting rgb/grayscale target image for 760x512 image dimension 
def write_circle_target_data1_3(data, filepath, figsize=(760, 512), draw_grid=False, tick_step=10):
    fig = go.Figure()
    for i in range(len(data)):
        for j in range(len(data[i])):
            fig.add_trace(go.Scatter(x=data[i][j]["y"],
                        y=data[i][j]["x"],
                        marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[0]/figsize[1]),    
                        #marker = {"size":np.array(data[i][j]["widths"])*np.sqrt(figsize[1]/figsize[0]),     
            "sizemode":'diameter',
            "sizeref": 0.87,
                                 },
                        ))
    
    fig.update_traces(mode='markers', marker_line_width=1, visible=False)

    fp_parts = filepath.rsplit(".", 1)
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=30, b=30),
        width=figsize[0], height=figsize[1],
        xaxis_range=[0,110],
        yaxis_range=[-0.5,10],
        yaxis={"showticklabels": False, "linewidth": 4, "linecolor": 'black'},
        xaxis={"showticklabels": False, "linewidth": 4, "linecolor": 'black', "ticks": "outside", "dtick": tick_step, "ticklen": 0, "tickwidth": 0, "ticklen": 10, "tickwidth": 4, "showgrid":False},
        showlegend=False
    )    
    fig.update_traces(mode='markers', marker_line_width=3.5, marker_color="white", marker_line_color="black", visible=True)
    if draw_grid:
    	fig.update_xaxes(showgrid=True, gridcolor='black', griddash='dash', gridwidth=3)
    pio.write_image(fig=fig, file=f"{fp_parts[0]}.{fp_parts[1]}", format="png", width=figsize[0], height=figsize[1])

    

# Convert data to numpy dataframe
def serialize_data(data, headers):
    df = {}
    
    for h in headers:
        df[h] = []
    
    for d in data:
        for h in d: 
            val_str = ""
            if h == "x":
                val_str +=  ", ".join(map(str, d[h]))
                val_str += " ."
            else:
                for vals in d[h]:
                    val_str +=  ", ".join(map(str, vals))
                    val_str += " ."
            df[h].append(val_str)
            
    return pd.DataFrame(df)
    
    

