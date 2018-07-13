import sys;  
from simple_navier_stokes import *; 
import numpy as np;
import argparse as argparse;
import matplotlib.pyplot as plt;
import time as time; 

startTime = time.time();

file_name = "data/cylinder/";
 
sns = sns.cylinder.cylinder (file_name);

parser = argparse.ArgumentParser(prog="python cylinder.py",
  description = "Initialises and configures a simple navier stokes solver for the 2D karman sheet.");

parser.add_argument('--finalTime', '-T', help='Final Time of the program.', type=float, default=0.01)
parser.add_argument('--numberSteps', '-N', help='Number of steps the program iterates.', type=int, default=1000)
parser.add_argument('--dynamicViscosity', '-V', help='Fluid dynamic viscosity.', type=float, default=sns.finalTime)
parser.add_argument('--density', '-D', help='Fluid mass density.', type=float, default=sns.density)
parser.add_argument('--meshFineness', '-M', help='Fineness of the mesh.', type=int, default=sns.meshFineness)
parser.add_argument('--visualization', '-S', help='Show visualization? Slows down excecution..', type=bool, default=True)
parser.add_argument('--animationInterval', '-A', help='How often the visualization is updated..', type=int, default=sns.animationInterval)
args = parser.parse_args();

sns.finalTime			= args.finalTime;
sns.numberTimeSteps	    = args.numberSteps;
sns.deltaTime 			= sns.finalTime / sns.numberTimeSteps;
sns.dynamicViscosity	= args.dynamicViscosity;
sns.density 			= args.density; 
sns.meshFineness 		=  args.meshFineness;
sns.visualization 		=  args.visualization;
sns.animationInterval 	=  args.animationInterval;
try:
	print "PreProcessing Simple NavierStokes. \n";
	sns.preProcess ();
	
	sns.show();
except AssertionError:
	print "Assertion Error was thrown. Likely, simple_navier_stokes was not preprocessed. \n";
except AttributeError:
	print "Attribute Error thrown. This is something related to closing the window; idc. \n";

finalTime = time.time ();

print "Process took %.3f seconds." % ( finalTime - startTime);