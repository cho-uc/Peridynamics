# Peridynamics
	
Benchmark model 05 =  Block of material under tranverse loading

OpenCl
v1 = Parallelize all loops
v2 = Parallel version with partial serialized
v3 = Parallel version with partial serialized, not write to CPU for each time step
v4 = Parallel version with partial serialized, not write to CPU for each time step
...
v5 = Parallel version with partial serialized, set critical time step as parallel
	