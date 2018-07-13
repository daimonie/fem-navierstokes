import sys;  
from simple_navier_stokes import simple_navier_stokes, cylinder; 
import numpy as np;
import argparse as argparse;
import matplotlib.pyplot as plt;
import time as time; 

start_time = time.time();

file_name = "data/cylinder/";
 
sns = cylinder.cylinder(file_name);

parser = argparse.ArgumentParser(prog="python cylinder.py",
  description = "Initialises and configures a simple navier stokes solver for the 2D karman sheet.");

parser.add_argument('--final_time', '-T', help='Final Time of the program.', type=float, default=0.01)
parser.add_argument('--numberSteps', '-N', help='Number of steps the program iterates.', type=int, default=1000)
parser.add_argument('--dynamicViscosity', '-V', help='Fluid dynamic viscosity.', type=float, default=sns.final_time)
parser.add_argument('--density', '-D', help='Fluid mass density.', type=float, default=sns.density)
parser.add_argument('--meshFineness', '-M', help='Fineness of the mesh.', type=int, default=sns.mesh_fineness)
parser.add_argument('--visualization', '-S', help='Show visualization? Slows down excecution..', type=bool, default=True)
parser.add_argument('--animationInterval', '-A', help='How often the visualization is updated..', type=int, default=sns.animation_interval)
args = parser.parse_args();

sns.final_time			= args.final_time;
sns.number_time_steps	= args.numberSteps;
sns.delta_time 			= args.final_time / args.numberSteps;
sns.dynamic_viscosity	= args.dynamicViscosity;
sns.density 			= args.density; 
sns.mesh_fineness 		= args.meshFineness;
sns.visualization 		= args.visualization;
sns.animation_interval 	= args.animationInterval;
try:
	print "PreProcessing Simple NavierStokes. \n";
	sns.pre_process ();
	
	sns.show();
except AssertionError:
	print "Assertion Error was thrown. Likely, simple_navier_stokes was not preprocessed. \n";
	plt.close()
	sys.exit()
except AttributeError:
	print "Attribute Error thrown. This is something related to closing the window; idc. \n";
	plt.close()
	sys.exit()
except TypeError:
	print "TypeError Error thrown. Probably means the code is wrong somehow. Bai. \n";
	plt.close()
	sys.exit()

final_time = time.time ();

print "Process took %.3f seconds." % ( final_time - start_time);