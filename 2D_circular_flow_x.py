#!/usr/bin/env python                                                                                                                                                
import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
import numpy as np
import dolfinx
from dolfinx.fem.assemble import assemble_matrix,assemble_scalar
from dolfinx.fem import FunctionSpace, Function, Constant,form
from dolfinx.plot import create_vtk_mesh
import pyvista


# Create mesh
gmsh.initialize()
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin",0.02)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.02)
gmsh.model.mesh.generate(gdim)
gmsh.write('circular_flow_x.msh')

# Interfacing with GMSH in DOLFINx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.FunctionSpace(domain, ("CG", 1))

# Defining a spatially varying load
x = ufl.SpatialCoordinate(domain)
miu = 10
dpdz = 10
p = 1/miu * dpdz


# Creating a Dirichlet b.c. using geometrical conditions
def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)
boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)

def boundary(x, on_boundary):
    return on_boundary

bc = fem.dirichletbc(ScalarType(0), boundary_dofs, V)

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Interpolation of a ufl-expression
#Q = fem.FunctionSpace(domain, ("CG", 5))
#expr = fem.Expression(p, Q.element.interpolation_points())
#pressure = fem.Function(Q)
#pressure.interpolate(expr)

print('area is')
one = Constant(domain, ScalarType(1))
f = form(one*ufl.dx)
area = assemble_scalar(f)
print(area)

print('mean velocity is')
f2 = form(uh*ufl.dx)
mv = assemble_scalar(f2)/area
print(mv)

topology, cell_types, x = create_vtk_mesh(domain, 2)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

grid.point_data["u"] = uh.x.array
warped = grid.warp_by_scalar("u", factor=10)
plotter = pyvista.Plotter()
plotter.background_color = "white"
plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False, scalars="u")

if not pyvista.OFF_SCREEN:
	plotter.show()
else:
	plotter.screenshot("deflection.png")


coor1 = V.tabulate_dof_coordinates()
np.savetxt('circular_coord.txt', coor1)
u1 = uh.x.array.real.astype(np.float32)
np.savetxt('circular_value.txt', u1)
