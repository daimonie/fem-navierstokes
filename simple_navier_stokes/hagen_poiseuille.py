import matplotlib  
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation 
from matplotlib.ticker import NullFormatter
from fenics import *
from mshr import *
import sys as sys
from simple_navier_stokes import *

class hagen_poiseuille(simple_navier_stokes):
    def __init__(self, file_name): 
        simple_navier_stokes.__init__(self, file_name)

        self.channel_width = 0.41;
        self.channel_length = 1.0; 
    def create_mesh(self):
        # Create mesh
        channel = Rectangle(Point(0, 0), Point(self.channel_length, self.channel_width)) 
        domain = channel

        self.mesh = generate_mesh(domain, self.mesh_fineness)

        self.reynolds_guess = self.reynolds(4.0*1.5/self.channel_width,  self.channel_width)


    def create_inflow_profile (self):
        self.inflow_profile_expression = ('4.0*1.5*x[1]*(%.3f - x[1]) / pow(%.3f, 2)' %(self.channel_width, self.channel_width), '0')
    def create_boundaries(self):
        boundary_inflow_expression   = 'near(x[0], 0)'
        boundary_walls_expression    = 'near(x[1], 0) || near(x[1], %.3f)' % (self.channel_width)
         
        boundary_outflow_expression  = 'near(x[0], %.3f)'% (self.channel_length)
 
        boundary_inflow = DirichletBC(self.velocity_function_space, Expression(self.inflow_profile_expression, degree=2), boundary_inflow_expression)
        
        boundary_walls = DirichletBC(self.velocity_function_space, Constant((0, 0)), boundary_walls_expression)
        boundary_outflow = DirichletBC(self.pressure_function_space, Constant(0), boundary_outflow_expression)

        self.boundary_conditions_velocity = [boundary_inflow, boundary_walls]
        self.boundary_conditions_pressure = [boundary_outflow];
 