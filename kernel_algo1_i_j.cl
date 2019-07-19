__kernel void algo1(const float small_delta, __global float *xi, \
					__global const float *delta_V, __global float *m,__global const float *x)
{ 
    
    int i = get_global_id(0);
	int j = get_global_id(1);
	
    m[i]=0;
	xi[i] = x[i]-x[j]; 			// xi as global var because cannot initialize array with size as var
    float omega=exp(-100.0/(small_delta*small_delta)); // include function?
	
	m[i] = m[i]+ (omega*delta_V[j]);
}