# Peridynamics

Extension of peridynamics software with GPU computing.

Benchmark model 01 = Longitudinal Vibration of a Bar

Benchmark model 02 = Bar under tension

	benchmark_02.cpp = neighbor list in 2D array
	
	benchmark_02_1d.cpp = neighbor list in 1D array
	
Benchmark model 05 =  Block of material under tranverse loading

	benchmark_05.cpp = neighbor list in 2D array
	
	benchmark_05_1d.cpp = neighbor list in 1D array
	
VTK_reader

	v01 : straight from source ReadLegacyUnstructuredGrid from VTK website with few changes
	
	v02 : Remove node numbering
	
	v03 : Add 2D text : unsuccessful yet
	
	ColoredPoints : Read from text file (not from VTK legacy format) 
	
	coloring according to displacement


# TODO
Optimize parallel version on OpenCl and Cuda
