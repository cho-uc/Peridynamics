__kernel void main_method(__global float *u_dot_x_nhalf, 
			__global float *u_dot_y_nhalf,
			__global float *u_dot_z_nhalf,
			__global float *u_dot_x_n0,
			__global float *u_dot_y_n0,
			__global float *u_dot_z_n0,
			
			__global const size_t *time_step)
{ 
    int i = get_global_id(0);
	//for (size_t j = 0; j < time_step; ++j){
	u_dot_x_nhalf[i]=u_dot_x_n0[i]+4.0;
	u_dot_y_nhalf[i]=u_dot_y_n0[i]+4.0;
	u_dot_z_nhalf[i]=u_dot_z_n0[i]+4.0;
	
	
}