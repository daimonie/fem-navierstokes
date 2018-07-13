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

		self.finalTime			= 5.0;
		self.numberTimeSteps	= 10000.0;
		self.deltaTime 			= self.finalTime / self.numberTimeSteps;
		self.dynamicViscosity	= 0.001;
		self.density 			= 1.0;

		self.meshFineness =  36;

		self.mesh = None;

		self.preProcessed = False;

		self.fileName = file_name;

		self.visualization = False;

 		self.animationInterval = 1; 
	def reynolds(self, velocity, lengthscale):
		return self.density * velocity * lengthscale / self.dynamicViscosity;
	def preProcess(self):

		if self.visualization:
			self.figure = plt.figure( figsize=(20,20));
		self.createMesh();
		self.createTrialTest();
		self.createInflowProfile ();
		self.createBoundaries();
		self.createInflowProfile ();
		self.createVariationalExpressions ();
		self.createFiles ();

		self.numberTimeSteps = int(self.numberTimeSteps);

		self.preProcessed = True;

	def createMesh(self):
		# Create mesh
		channel = Rectangle(Point(0, 0), Point(2.2, 0.41));
		cylinder = Circle(Point(0.2, 0.2), 0.05);


		domain = channel - cylinder;
		
		self.mesh = generate_mesh(domain, self.meshFineness);
	def createBoundaries(self):
		boundaryInflowExpression   = 'near(x[0], 0)';
		boundaryWallsExpression    = 'near(x[1], 0) || near(x[1], 0.41)';
		boundaryCylinderExpression = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3';

		boundaryOutflowExpression  = 'near(x[0], 2.2)';

		boundaryInflow = DirichletBC(self.velocityFunctionSpace, Expression(self.inflowProfileExpression, degree=2), boundaryInflowExpression)
		boundaryWalls = DirichletBC(self.velocityFunctionSpace, Constant((0, 0)), boundaryWallsExpression)
		boundaryCylinder = DirichletBC(self.velocityFunctionSpace, Constant((0, 0)), boundaryCylinderExpression)
		boundaryOutflow = DirichletBC(self.pressureFunctionSpace, Constant(0), boundaryOutflowExpression)


		self.boundaryConditionsVelocity = [boundaryInflow, boundaryWalls, boundaryCylinder];
		self.boundaryConditionsPressure = [boundaryOutflow];

		self.boundaryConditionsTest = 'near(x[0], 0)';
	def createInflowProfile (self):
		self.inflowProfileExpression = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0');
	def createTrialTest(self):

		self.velocityFunctionSpace = VectorFunctionSpace(self.mesh, 'P', 2)
		self.pressureFunctionSpace = FunctionSpace(self.mesh, 'P', 1)

		self.velocityTrial = TrialFunction (self.velocityFunctionSpace); # u
		self.velocityTest = TestFunction(self.velocityFunctionSpace); # v

		self.previousVelocity = Function(self.velocityFunctionSpace); # u_n
		self.currentVelocity = Function(self.velocityFunctionSpace); # u_

		self.pressureTrial = TrialFunction(self.pressureFunctionSpace); #p
		self.pressureTest = TestFunction(self.pressureFunctionSpace); #q 

		self.previousPressure = Function(self.pressureFunctionSpace); # p_n
		self.currentPressure =  Function(self.pressureFunctionSpace); # p_

	def createVariationalExpressions(self):
		# Define expressions used in variational forms
		# this just makes "F1" readable for a physicist.
		v =  self.velocityTest;

		u = self.velocityTrial;
		u_n = self.previousVelocity;
		u_ = self.currentVelocity;

		q = self.pressureTest;

		p = self.pressureTrial;
		p_n = self.previousPressure;
		p_ = self.currentPressure;


		U  = 0.5*(u + u_n);
		n  = FacetNormal(self.mesh);
		f  = Constant((0, 0));
		k  = Constant(self.deltaTime);
		mu = Constant(self.dynamicViscosity);
		rho = Constant(self.density);

		epsilon = lambda uu: sym(nabla_grad(uu));
		sigma = lambda uu, pp: 2 * mu * epsilon(uu) - pp * Identity(len(uu));


		F1 = rho*dot((u - u_n) / k, v)*dx \
		   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
		   + inner(sigma(U, p_n), epsilon(v))*dx \
		   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
		   - dot(f, v)*dx;

		self.tentativeVariationalProblemLeft = lhs(F1);
		self.tentativeVariationalProblemRight = rhs(F1);
		self.tentativeVariationalProblemMatrix = assemble(self.tentativeVariationalProblemLeft);

		# Define variational problem for step 2
		self.pressureCorrectionVariationalProblemLeft = dot(nabla_grad(p), nabla_grad(q))*dx
		self.pressureCorrectionVariationalProblemRight = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
		self.pressureCorrectionVariationalProblemMatrix = assemble(self.pressureCorrectionVariationalProblemLeft);

		# Define variational problem for step 3
		self.velocityCorrectionVariationalProblemLeft = dot(u, v)*dx
		self.velocityCorrectionVariationalProblemRight = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
		self.velocityCorrectionVariationalProblemMatrix = assemble(self.velocityCorrectionVariationalProblemLeft);
		
		#Apply boundary conditions
		[bc.apply(self.tentativeVariationalProblemMatrix) for bc in self.boundaryConditionsVelocity];
		[bc.apply(self.pressureCorrectionVariationalProblemMatrix) for bc in self.boundaryConditionsPressure];
	def createFiles(self):
		# Create XDMF files for visualization output
		self.velocityXFile = XDMFFile('%s/velocity.xdmf' % self.fileName)
		self.pressureXFile = XDMFFile('%s/pressure.xdmf' % self.fileName)

		# Create time series (for use in reaction_system.py)
		self.velocityTimeSeries = TimeSeries('%s/velocity_series' % self.fileName)
		self.pressureTimeSeries = TimeSeries('%s/pressure_series' % self.fileName)

		#save mesh to file
		MeshFile = '%s/cylinder.xml.gz' % self.fileName;
		# not permitted in parallel
		File(MeshFile) << self.mesh
 

	def tentativeVelocityStep(self):
		boundary = assemble(self.tentativeVariationalProblemRight);
		[bc.apply(boundary) for bc in self.boundaryConditionsVelocity]; 

		solve(self.tentativeVariationalProblemMatrix, self.currentVelocity.vector(), boundary, 'bicgstab', 'hypre_amg');
	def pressureCorrectionStep(self):
		boundary = assemble(self.pressureCorrectionVariationalProblemRight);
		[bc.apply(boundary) for bc in self.boundaryConditionsPressure];

		solve(self.pressureCorrectionVariationalProblemMatrix, self.currentPressure.vector(), boundary, 'bicgstab', 'hypre_amg');
	def velocityCorrectionStep(self):
		boundary =  assemble(self.velocityCorrectionVariationalProblemRight);
		solve(self.velocityCorrectionVariationalProblemMatrix, self.currentVelocity.vector(), boundary, 'cg', 'sor');
	def evolve(self, iteration): 
		self.time = iteration * self.deltaTime;

		if self.time > self.finalTime:
			plt.close ();
			return;

		self.tentativeVelocityStep ();
		self.pressureCorrectionStep ();
		self.velocityCorrectionStep ();

		# plot solution
		report = "t=%.9f, %d/%d" % (self.time, iteration, self.numberTimeSteps);

		if self.visualization: 
			if np.mod(iteration, self.animationInterval) == 0: 
				plt.subplot(311);
				plot(self.currentVelocity, title='Velocity %s'% report);
				plt.subplot(312);
				plot(self.currentPressure, title='Pressure %s'% report);
				plt.subplot(313);
				plot(self.mesh, title = 'Mesh');

		print "%s.\n" % report;

		#save to file 
		# Save solution to file (XDMF/HDF5)
		self.velocityXFile.write(self.currentVelocity, self.time);
		self.pressureXFile.write(self.currentPressure, self.time);

		# Save nodal values to file
		self.velocityTimeSeries.store(self.currentVelocity.vector(), self.time);
		self.pressureTimeSeries.store(self.currentPressure.vector(), self.time);
		#assign previous

		self.previousVelocity.assign( self.currentVelocity );
		self.previousPressure.assign( self.currentPressure ); 

	def show(self):
		""" Calls plt.show() """
		assert(self.preProcessed == True);
		if self.visualization:
			plt.plot([], [], 'r-')
			self.ani = animation.FuncAnimation(self.figure, self.evolve, interval=self.animationInterval) 
			plt.show ();
		else:
			for iteration in range(0, self.numberTimeSteps):
				self.evolve(iteration);