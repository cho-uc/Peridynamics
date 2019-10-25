# Peridynamics
	
Benchmark model 05 =  Block of material under tranverse loading

Cuda
v1 = Parallelize all loops
v2 = Parallel result with partial serialized (calloc)
v3 = Parallel result with partial serialized & Pinned memory
v4 = Parallel result with partial serialized, use new for array
v5 = Parallel result with partial serialized (calloc),  parallel critical_time_step

Discovery:
- Use CUDA host memory-pinned do not improve performance
- Use math functions in floating precision almost has no effect on performance
- Use new float array initialization is faster than calloc/malloc




