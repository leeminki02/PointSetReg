""" tools.py
This file contains helper functions for the point set registration.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_cloud(points, colors, show_eyes=True):
    # plot the data
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(-points[0], points[1], points[2], marker='x', color=colors)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    if show_eyes:
        ax.scatter((-0.06, 0.06), (0,0), (0,0), marker='o', color='r')
        ax.scatter((0),(0),(0), marker='P', color='k')
        ax.plot((-0.06, 0.06), (0,0), (0,0), color='r')
        ax.plot((0,0), (-0.1, 0.05), (0,0), color='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.view_init(120,-45,45)
    plt.show()

def plot_clouds(cloud1, cloud2, cloud1_color, cloud2_color, savefig = False, file_id = ''):
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    ax1.scatter(-cloud1[0], cloud1[1], cloud1[2], marker='.', color=cloud1_color)
    ax2.scatter(-cloud2[0], cloud2[1], cloud2[2], marker='.', color=cloud2_color)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax1.set_title('cloud1')
    ax2.set_title('cloud2')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_aspect('equal')
    ax1.view_init(120,-45,45)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_aspect('equal')
    ax2.view_init(120,-45,45)
    if savefig:
        plt.savefig('results/clouds_'+file_id+'.png')
    plt.show()

def make_transformation(points, colors, noiseThrsh=0.005, noisePoints=30):
    cloud2 = points.copy()
    cloud2_color = colors.copy()

    if (noiseThrsh > 0):
        # make some noise (less than 0.01) to cloud2 point data
        noise = np.random.normal(0, 0.005, cloud2.shape)
        cloud2 += noise

    if (noisePoints > 0):
        # add some random noise points to cloud2
        num_noise_points = 30
        # TODO: make random noise points more REALISTIC
        # make noise points in the range of cloud2
        noise_points = np.random.rand(3, num_noise_points) * (np.max(cloud2, axis=1) - np.min(cloud2, axis=1)).reshape(-1,1) + np.min(cloud2, axis=1).reshape(-1,1)
        # noise_points = np.random.rand(3, num_noise_points)
        cloud2 = np.concatenate((cloud2, noise_points), axis=1)
        # make color data for noise points. but to make them visible, set the colors to pure red.
        noise_colors = np.array([[1, 0, 0]]*num_noise_points) # color red [1, 0, 0] to noise points
        cloud2_color = np.concatenate((cloud2_color, noise_colors), axis=0)
    
    # generate transformations: translation & rotation
    # translation
    t = np.array([0.1, 0.1, 0.2])
    # rotation
    theta = np.random.random() * np.pi/2
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    # scaling: random between 0.5~2.0
    s = np.random.random()*1.5 + 0.5
    
    # apply transformation to cloud2
    cloud2 = s * np.dot(R, cloud2) + t.reshape(-1,1)
    return cloud2, cloud2_color, t, R, s
