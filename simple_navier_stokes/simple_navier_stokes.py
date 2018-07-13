import matplotlib 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation 
from matplotlib.ticker import NullFormatter
from fenics import *;
from mshr import *;

class simple_navier_stokes (object):
    """simple_navier_stokes is an implementation of the ft08 fenics tutorial script.
    It encapsulates the script as an animation object, such that matplotlib displays
    the realtime calculations. In the future, this should just be done by inheriting
    some kind of animationObject superclass."""
    def __init__(self, file_name):
        """ Initialise matplotlib functions, set some parameters"""

        self.final_time         = 5.0;
        self.number_time_steps  = 10000.0;
        self.delta_time         = self.final_time / self.number_time_steps;
        self.dynamic_viscosity  = 0.001;
        self.density            = 1.0;

        self.mesh_fineness =  36;

        self.mesh = None;

        self.pre_processed = False;

        self.file_name = file_name;

        self.visualization = False;

        self.animation_interval = 1; 

        self.reynolds_guess = -1.0

    def reynolds(self, velocity, lengthscale):
        return self.density * velocity * lengthscale / self.dynamic_viscosity;
    def pre_process(self):

        if self.visualization:
            self.figure = plt.figure( figsize=(20,20));
        self.create_mesh();
        self.create_trial_test();
        self.create_inflow_profile ();
        self.create_boundaries();
        self.create_inflow_profile ();
        self.create_variational_expressions (); 

        self.number_time_steps = int(self.number_time_steps);

        self.pre_processed = True;

    def create_mesh(self):
        # Create mesh
        channel = Rectangle(Point(0, 0), Point(2.2, 0.41)); 

        domain = channel;
        
        self.mesh = generate_mesh(domain, self.mesh_fineness);
    def create_inflow_profile (self):
        self.inflow_profile_expression = ('5', '0');

    def create_boundaries(self):
        boundary_inflow_expression   = 'near(x[0], 0)';
        boundary_walls_expression    = 'near(x[1], 0) || near(x[1], 0.41)'; 

        boundary_outflow_expression  = 'near(x[0], 2.2)';

        boundary_inflow = DirichletBC(self.velocity_function_space, Expression(self.inflow_profile_expression, degree=2), boundary_inflow_expression)
        boundary_walls = DirichletBC(self.velocity_function_space, Constant((0, 0)), boundary_walls_expression)
        boundary_outflow = DirichletBC(self.pressure_function_space, Constant(0), boundary_outflow_expression)


        self.boundary_conditions_velocity = [boundary_inflow, boundary_walls];
        self.boundary_conditions_pressure = [boundary_outflow];

        self.boundary_conditions_test = 'near(x[0], 0)';
    def create_trial_test(self):

        self.velocity_function_space = VectorFunctionSpace(self.mesh, 'P', 2)
        self.pressure_function_space = FunctionSpace(self.mesh, 'P', 1)

        self.velocity_trial = TrialFunction (self.velocity_function_space); # u
        self.velocity_test = TestFunction(self.velocity_function_space); # v

        self.previous_velocity = Function(self.velocity_function_space); # u_n
        self.current_velocity = Function(self.velocity_function_space); # u_

        self.pressure_trial = TrialFunction(self.pressure_function_space); #p
        self.pressure_test = TestFunction(self.pressure_function_space); #q 

        self.previous_pressure = Function(self.pressure_function_space); # p_n
        self.current_pressure =  Function(self.pressure_function_space); # p_

    def create_variational_expressions(self):
        # Define _expressions used in variational forms
        # this just makes "F1" readable for a physicist.
        v =  self.velocity_test;

        u = self.velocity_trial;
        u_n = self.previous_velocity;
        u_ = self.current_velocity;

        q = self.pressure_test;

        p = self.pressure_trial;
        p_n = self.previous_pressure;
        p_ = self.current_pressure;


        U  = 0.5*(u + u_n);
        n  = FacetNormal(self.mesh);
        f  = Constant((0, 0));
        k  = Constant(self.delta_time);
        mu = Constant(self.dynamic_viscosity);
        rho = Constant(self.density);

        epsilon = lambda uu: sym(nabla_grad(uu));
        sigma = lambda uu, pp: 2 * mu * epsilon(uu) - pp * Identity(len(uu));


        F1 = rho*dot((u - u_n) / k, v)*dx \
           + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
           + inner(sigma(U, p_n), epsilon(v))*dx \
           + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
           - dot(f, v)*dx;

        self.tentative_variational_problem_left = lhs(F1);
        self.tentative_variational_problem_right = rhs(F1);
        self.tentative_variational_problem_matrix = assemble(self.tentative_variational_problem_left);

        # Define variational problem for step 2
        self.pressure_correction_variational_problem_left = dot(nabla_grad(p), nabla_grad(q))*dx
        self.pressure_correction_variational_problem_right = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
        self.pressure_correction_variational_problem_matrix = assemble(self.pressure_correction_variational_problem_left);

        # Define variational problem for step 3
        self.velocity_correction_variational_problem_left = dot(u, v)*dx
        self.velocity_correction_variational_problem_right = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
        self.velocity_correction_variational_problem_matrix = assemble(self.velocity_correction_variational_problem_left);
        
        #Apply boundary conditions
        [bc.apply(self.tentative_variational_problem_matrix) for bc in self.boundary_conditions_velocity];
        [bc.apply(self.pressure_correction_variational_problem_matrix) for bc in self.boundary_conditions_pressure];
 
    def tentative_velocity_step(self):
        boundary = assemble(self.tentative_variational_problem_right);
        [bc.apply(boundary) for bc in self.boundary_conditions_velocity]; 

        solve(self.tentative_variational_problem_matrix, self.current_velocity.vector(), boundary, 'bicgstab', 'hypre_amg');
    def pressure_correction_step(self):
        boundary = assemble(self.pressure_correction_variational_problem_right);
        [bc.apply(boundary) for bc in self.boundary_conditions_pressure];

        solve(self.pressure_correction_variational_problem_matrix, self.current_pressure.vector(), boundary, 'bicgstab', 'hypre_amg');
    def velocity_correction_step(self):
        boundary =  assemble(self.velocity_correction_variational_problem_right);
        solve(self.velocity_correction_variational_problem_matrix, self.current_velocity.vector(), boundary, 'cg', 'sor');
    def evolve(self, iteration): 
        self.time = iteration * self.delta_time;

        if self.time > self.final_time:
            plt.close ();
            return;

        self.tentative_velocity_step ();
        self.pressure_correction_step ();
        self.velocity_correction_step ();

        # plot solution
        report = "t=%.9f, %d/%d" % (self.time, iteration, self.number_time_steps);

        if np.mod(iteration, self.animation_interval) == 0: 
            plt.subplot(311);
            plot(self.current_velocity, title='Velocity %s at Re=%d'% (report, self.reynolds_guess));
            plt.subplot(312);
            plot(self.current_pressure, title='Pressure %sat Re=%d'% (report, self.reynolds_guess));
            plt.subplot(313);
            plot(self.mesh, title = 'Mesh'); 

        print "%s.\n" % report; 
        #assign previous

        self.previous_velocity.assign( self.current_velocity );
        self.previous_pressure.assign( self.current_pressure ); 

    def show(self):
        """ Calls plt.show() """
        assert(self.pre_processed == True);
        plt.plot([], [], 'r-')
        plt.clf(); 
        if self.visualization:
            plt.clf()
            self.ani = animation.FuncAnimation(self.figure, self.evolve, interval=self.animation_interval) 
            plt.show ();
        else:
            for iteration in range(0, self.number_time_steps):
                self.evolve(iteration);
                plt.savefig("%s/Reynolds%dIteration%d.png" % (self.file_name, self.reynolds_guess, iteration));
                plt.close();