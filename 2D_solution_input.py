#!/usr/bin/env python

import pyvista
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
         form, locate_dofs_geometrical, locate_dofs_topological,assemble_matrix)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
 dx, grad, dot)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import meshio

from image2gmsh_model_new import load_image, image2gmsh_model

import gmsh
import numpy as np
from scipy.interpolate import griddata
from PIL import Image

import sys


def plotMesh(mesh, uh):
    topology, cell_types, x = create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.point_data["u"] = uh.x.array
    warped = grid.warp_by_scalar("u", factor= 0.25 )
    plotter = pyvista.Plotter()
    plotter.background_color = "white"
    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False, scalars="u")
    plotter.show_bounds()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("deflection.png") 


 
def generateMesh(file_name):

    print('running computeMesh function')
    # Subdomain defined from external mesh data
    gdim = 2
    gmsh_model_rank = 0
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(file_name, MPI.COMM_WORLD, gdim=2)
    V = FunctionSpace(mesh, ("CG", 1))
    
    # Defining a random spatially varying load
    p = 10

    # Find Area
    print('area is')
    one = Constant(mesh, ScalarType(1))
    f = form(one*dx)
    area = assemble_scalar(f)
    print(area)
    
    
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh
    
    if gmsh_model_rank == 0:
        # Read in mesh
        msh = meshio.read(file_name)

        # Create and save one file for the mesh, and one file for the facets 
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)
    
    
   
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")
    
    u_bc = Function(V)
    walls_facets= ft.find(1)
    walls_dofs = locate_dofs_topological(V, mesh.topology.dim-1, walls_facets)
    bc = [dirichletbc(ScalarType(0), walls_dofs, V)]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = p * v * dx
    problem = LinearProblem(a, L, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()


    # Find Average Velocity
    print('average velocity is')
    f2 = form(uh*dx)
    average_velocity = assemble_scalar(f2)/area
    print(average_velocity)
    
    # Return Values
    return uh, area, average_velocity, mesh, V
    


def combineMesh(V1, V2, uh_1, uh_2):

    coor1 = V1.tabulate_dof_coordinates()
    u1 = uh_1.x.array.real.astype(np.float32)
    coor2 = V2.tabulate_dof_coordinates()
    u2 = uh_2.x.array.real.astype(np.float32)

    coordinates = np.vstack((coor1, coor2))
    vertex_values = np.concatenate((u1, u2))
    
    np.savetxt('coord.txt', coordinates)
    np.savetxt('vert.txt', vertex_values)
    
    #print(coordinates[:,:2])

    # Interpolate velocity onto zero domain
    x = np.linspace(-0.5, 0.5, 256)
    y = np.linspace(-0.5, 0.5, 256)
    grid_x, grid_y = np.meshgrid(x,y)
    #vertex_values = np.uint8(vertex_values/np.max(vertex_values) * 255)
    grid = griddata(coordinates[:,:2], vertex_values, (grid_x, grid_y), method='linear', fill_value=0.0)
    grid/= np.max(grid)
    grid *= 255
    grid = grid.astype(np.uint8)
    img = Image.fromarray(grid, 'L')
    img.save('inlet_flow_profile.png')
                                                                                                                             
    

if __name__ == '__main__':
   
    img_fname = sys.argv[1]
    img = load_image(img_fname)
    flow_rate_ratio = 0.5  # V_in / V_out   or V_1 / V_2
    image2gmsh_model(img)
    
    uh_o1, A1, u1, m1, V1 = generateMesh("inner_contour_mesh.msh")
    uh_o2, A2, u2, m2, V2 = generateMesh("outer_contour_mesh.msh")
    
    u1_target = flow_rate_ratio / (A1 * (1 + flow_rate_ratio))
    u2_target = 1 / (A2 * (1 + flow_rate_ratio))
    
    print('target average inner velocity is')
    print(u1_target)
    print('target average outer velocity is')
    print(u2_target)
   
    uh_1 = Function(V1)
    uh_1.x.array[:] = uh_o1.x.array[:] / u1 * u1_target
    uh_2 = Function(V2)
    uh_2.x.array[:] = uh_o2.x.array[:] / u2 * u2_target
    
    print(type(uh_2.x))
    print(type(uh_2.x.array))
    print(type(uh_2.x.array[1]))
    
    
    
    plotMesh(m1, uh_1)
    plotMesh(m2, uh_2)
    
   
    
    #combineMesh(V1, V2, uh_1, uh_2)
    
    
    


