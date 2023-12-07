# 2D_flow_profile_solver

2D_flow_profile_solver is a 2D finite element solver to simulate fully developed laminar flow through arbitrarily shaped microchannels. The solver takes an image of the inlet boundary geometry and a flow rate ratio of the inner and outer flow as inputs, creating the velocity profile of the fully developed laminar low as output. 

# How to set up and install dependencies (e.g., FENicSx, numpy.)
  Prerequisites
  Installations

# How to use 

We can run two simple test cases to validate this finite element solver. 
```
python 2D_circular_flow_x.py
python 2D_square_input.py
```
The results of running these test cases are 3 files containing the information of the velocity value, coordinates, and a 3D pyvista plot for visualization, which could be compared with the theoretical solution of the fully developed laminar velocity profile of uniform flow passing through a circular or square channel

To find the fully developed velocity profile from the input image in Plus_final.png, use the following command line:
```
python 2D_solution_input.py PlusF_final.png
```
The solver first extracts the contours from the PlusF_final.png and then generates inner and outer meshes from the counters. The meshes would become the domain of this variational problem to solve the Navier-Stokes Equation in FENicSx.This line would generate a PNG file called inlet_flow_profile.png. Two text files will be automatically created: coord.txt, recording the coordination of the meshes, and vert.txt, writing down the vertex value on each mesh point. 

The flow rate ratio (inner flow rate / outer flow rate) is 0.5 by default. You can change that value in line 147
```
147    flow_rate_ratio = 0.5  # V_in / V_out   or V_1 / V_2
```



# Authors
* Yulin Zhou
* Dr. Dan Stoecklein

# Acknowledgements

