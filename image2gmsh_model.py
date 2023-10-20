#!/usr/bin/env python#!/usr/bin/env python

import gmsh
import numpy as np
from skimage import measure
from skimage import io
import skimage as sk
import scipy.ndimage as ndimage
import math
import sys
import os
import time
from rdp import rdp

from matplotlib import pyplot as plt


import pyvista
from dolfinx import plot
from dolfinx.io import gmshio
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI





theta_thresh = 10.0


def load_image(img_fname):
    print('Loading image {}'.format(img_fname))
    img = io.imread(img_fname)

    print(img.shape)
    if (len(img.shape) == 2):
        gray_img = img
    else:
        if (img.shape[2] == 3):
            gray_img = sk.color.rgb2gray(img)
        if (img.shape[2] == 3):
            rgb_img = sk.color.rgba2rbg(img)
            gray_img = sk.color.rgb2gray(rgb_img)

    '''
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(gray_img)
    plt.show()
    '''
    
    return gray_img
  
def get_contours(gray_img):
    height, width = gray_img.shape    
    # Normalize and flip (for some reason)
    raw_contours = sk.measure.find_contours(gray_img, 0.5)
 
    print('Found {} contours'.format(len(raw_contours)))

    contours = []
    for n, contour in enumerate(raw_contours):
        # Create an empty image to store the masked array
        r_mask = np.zeros_like(gray_img, dtype = int)  # original np.int
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        r_mask = ndimage.binary_fill_holes(r_mask)

        contour_area = float(np.count_nonzero(r_mask))/(float(height * width))
        print(np.count_nonzero(r_mask))
        if (contour_area >= 0.05):
            contours.append(contour)

    print('Reduced to {} contours'.format(len(contours)))


    '''
    num_contours = len(contours)
    fig, ax = plt.subplots(1,num_contours+1)
    ax[0].imshow(gray_img)
    for n in range(num_contours):
        ax[n+1].plot(contours[n][:,1], contours[n][:,0])
        ax[n+1].set_xlim([0, width])
        ax[n+1].set_ylim([0, height])


    fig, ax = plt.subplots(1,2)
    ax[0].imshow(gray_img)
    for n in range(num_contours):
        ax[0].plot(contours[n][:,1], contours[n][:,0])

    #ax[2].plot(contours[1][:,1], contours[1][:,0])
    #ax[2].plot(contours[1][:,1], contours[1][:,0])
    plt.show()
    '''
    

    for n, contour in enumerate(contours):
        contour[:,1] -= 0.5 * height
        contour[:,1] /= height

        contour[:,0] -= 0.5 * width
        contour[:,0] /= width
        contour[:,0] *= -1.0

    print("{:d} Contours detected".format(len(contours)))

    return contours


def optimize_contour(contour):
    print("Optimizing contour.")
    dir_flag = 0
    dir_bank = []

    contour_keep = []

    ## Use low-pass fft to smooth out 
    x = contour[:,1]
    #print('x')
    #print(x)
    #print('y')
    #y = 1j*contour[:,0]
    y = contour[:,0]

    #print(y)
    signal = x + 1j*y
    #print(signal)

    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    cutoff = 0.12
    fft[np.abs(freq) > cutoff] = 0 

    signal_filt = np.fft.ifft(fft)

    contour[:,1] = signal_filt.real
    contour[:,0] = signal_filt.imag

    #contour = rdp(contour)
    contour = rdp(contour, epsilon=0.0005)

    # Remove final point in RDP, which coincides with
    # the first point
    contour = np.delete(contour, len(contour)-1, 0)

    # cutoff of 0.15, eps of 0.005 works for inner flow

    #contour = reverse_opt_pass(contour)
    # Figure out a reasonable radius
    max_x = max(contour[:,1])
    min_x = min(contour[:,1])
    
    max_y = max(contour[:,0])
    min_y = min(contour[:,0])
    
    # Set characteristic lengths, epsilon cutoff
    lc = min((max_x - min_x), (max_y - min_y))
    mesh_lc = 0.01 * lc    

    return [contour, mesh_lc]




### nothing changes above




def outer_contour_to_gmsh(contour, mesh_lc, p_idx=1, l_idx=1, loop_idx=1):
    print('Running outer_contour_to_gmsh')
    line_init = l_idx
    g = gmsh.model.geo
    gmsh.initialize()
    gmsh.model.add("outer_contour_mesh")
    lc = mesh_lc
    g.addPoint(-0.5, -0.5, 0, lc, p_idx)
    g.addPoint( 0.5, -0.5, 0, lc, p_idx + 1)
    g.addPoint( 0.5,  0.5, 0, lc, p_idx + 2)
    g.addPoint(-0.5,  0.5, 0, lc, p_idx + 3)
    
    g.addLine(p_idx, p_idx + 1, l_idx)
    g.addLine(p_idx + 1, p_idx + 2, l_idx + 1)
    g.addLine(p_idx + 2, p_idx + 3, l_idx + 2)
    g.addLine(p_idx + 3, p_idx, l_idx + 3)
    g.addCurveLoop(list(range(l_idx , l_idx + 4)), loop_idx)
    
    p_idx += 3
    p_idx_closure = p_idx + 1
    for point in contour:
        p_idx += 1
        g.addPoint(point[1], point[0], 0, lc, p_idx)
    
    l_idx += 3
    for line in range(len(contour)-1):
        l_idx += 1
        g.addLine(l_idx, l_idx + 1, l_idx)
        
    g.addLine(l_idx + 1, p_idx_closure, l_idx + 1)
    g.addCurveLoop(list(range(p_idx_closure , l_idx + 2)), 2)
    g.addPlaneSurface([1,2], 1)
    g.synchronize()
    g.addPhysicalGroup(1, list(range(line_init , l_idx + 2)), 1)
    g.addPhysicalGroup(2, [1], name = "outer_surface") 
    
    gmsh.model.mesh.generate(2)
    gmsh.write("outer_contour_mesh.msh")
    if '-nopopup' not in sys.argv:
    	gmsh.fltk.run()
    return gmsh.model



def inner_contour_to_gmsh(contour, mesh_lc):

    print('Running inner_contour_to_gmsh')
    
    gmsh.initialize()
    gmsh.model.add("inner_contour_mesh")
    g = gmsh.model.geo
    lc = mesh_lc
    idx = 0
    for point in contour:
        idx += 1
        g.addPoint(point[1], point[0], 0, lc, idx)
    idx = 0
    for line in range(len(contour)-1):
        idx += 1
        g.addLine(idx, idx + 1, idx)
    g.addLine(idx + 1, 1, idx + 1)
    g.addCurveLoop(list(range(1 , idx + 2)), 1)
    g.addPlaneSurface([1], 1)
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, list(range(1 , idx + 2)), 1)
    gmsh.model.addPhysicalGroup(2, [1], name = "inner_surface") 
    
    gmsh.model.mesh.generate(2)
    gmsh.write("inner_contour_mesh.msh")
    
    
    
    return gmsh.model
    
    
    
def process_2_channel_mesh_model(contours):
    contour_inner = contours[1]
    contour_outer = contours[0]

    [contour_inner, mesh_lc_a] = optimize_contour(contour_inner)
    [contour_outer, mesh_lc_b] = optimize_contour(contour_outer)

    inner_model = inner_contour_to_gmsh(contour_inner, mesh_lc_a)
    outer_model = outer_contour_to_gmsh(contour_outer, mesh_lc_a)
    return inner_model, outer_model


def image2gmsh_model(img):
    print('Running image2gmsh_model')
    contours = get_contours(img)   
    inner_model, outer_model = process_2_channel_mesh_model(contours)
    #process_2_channel_geo_3D_extrude(job_name, contours)        
    return inner_model, outer_model

def main():
    img_fname = sys.argv[1]
    img = load_image(img_fname)
    inner_model, outer_model = image2gmsh_model(img)    
    

if __name__ == '__main__':
    main()
    
    
