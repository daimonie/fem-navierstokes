import matplotlib  
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation 
from matplotlib.ticker import NullFormatter
from fenics import *;
from mshr import *;
import sys as sys;

class cylinder(simple_navier_stokes):
    def __init__(self, file_name): 
        simple_navier_stokes.__init__(self, file_name);

    def create_mesh(self):
        # Create mesh
        channel = Rectangle(Point(0, 0), Point(2.2, 0.41));
        cylinder = Circle(Point(0.2, 0.2), 0.05);


        domain = channel - cylinder;

        self.mesh = generate_mesh(domain, self.mesh_fineness);
    def create_inflow_profile (self):
        self.inflow_profile_expression = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0');
    def create_boundaries(self):
        boundary_inflow_expression   = 'near(x[0], 0)';
        boundary_walls_expression    = 'near(x[1], 0) || near(x[1], 0.41)';
        boundary_cylinder_expression = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3';

        boundary_outflow_expression  = 'near(x[0], 2.2)';

        boundary_inflow = DirichletBC(self.velocity_function_space, Expression(self.inflow_profile_expression, degree=2), boundary_inflow_expression)
        boundary_walls = DirichletBC(self.velocity_function_space, Constant((0, 0)), boundary_walls_expression)
        boundary_cylinder = DirichletBC(self.velocity_function_space, Constant((0, 0)), boundary_cylinder_expression)
        boundary_outflow = DirichletBC(self.pressure_function_space, Constant(0), boundary_outflow_expression)


        self.boundary_conditions_velocity = [boundary_inflow, boundary_walls, boundary_cylinder];
        self.boundary_conditions_pressure = [boundary_outflow];

        self.boundary_conditions_test = 'near(x[0], 0)';