//Cuda kernel for Benchmark 05

#include <math.h>       //exp, pi
#define M_PI 3.14159265358979323846

__global__ void discretize_blocks( const float *delta_x,
	 const float *delta_y,
	 const float *delta_z,
	 float *x,  float *y,  float *z, float *delta_V,
	 const size_t *ndivx,
	 const size_t *ndivy,
	 const size_t *ndivz){ 
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;

	x[i]=(2.0*floor((float)i/(float)(*ndivy)/(float)(*ndivz)) +1.0)  *(float)(*delta_x)/2.0;
	y[i]=(2*(floor((float)i/ (float)(*ndivz)) - (float)(*ndivy)*floor((float)i/(float)(*ndivy)/(float)(*ndivz))) + 1.0) \
		* (float)(*delta_y)/2.0;
	z[i]= (2.0*(i%(size_t)(*ndivz)) +1.0) * (float)(*delta_z)/2.0;
	delta_V[i]=(float)(*delta_x)*(float)(*delta_y)*(float)(*delta_z);
	
}

__global__ void critical_time_step(size_t *neighbor_list_pointer,
	 size_t *neighbor_list,
	 float *delta_V,
	 float *x,	 float *y,  float *z, /*6*/
	 float *V_dot_C_array,
	 size_t *iter_neighbor_list,
	 const size_t *node,
	 const float *small_delta,
	 const float *k_bulk_mod){

	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	//__local float V_dot_C_array_device[(int)(*node)]; //variable length is not supported
	//__local float V_dot_C_array_device[30000];
	
	float V_dot_C_temp=0.0; //Re-initialization
			
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		float C_p=18.0*(float)(*k_bulk_mod)/(sqrt(xi_square)*M_PI*pow((float)(*small_delta),(float)4.0));
		V_dot_C_temp += (C_p*delta_V[j]);
	}
	V_dot_C_array[i]=V_dot_C_temp;

	/*V_dot_C_array_device[i]=V_dot_C_temp;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if(V_dot_C_temp_array_device[i]>V_dot_C){ //find max
		V_dot_C=V_dot_C_temp;
	}*/
	
}

float critical_time_step_parallel(size_t *buffer_neighbor_list_pointer,
		size_t *buffer_neighbor_list, 
		float *buffer_delta_V, 
		float *buffer_x, float *buffer_y, float *buffer_z,
		size_t* buffer_iter_neighbor_list, 
		size_t* buffer_node, float * buffer_small_delta, float *buffer_k_bulk_mod, 
		const size_t * node, const float* ro, int *gridSize, int* blockSize){
	
	float *V_dot_C_array = new float[(size_t)(*node)];
	float *buffer_V_dot_C_array;
	cudaMalloc(&buffer_V_dot_C_array, (size_t)(*node)*sizeof(float));
	
	critical_time_step<<<(int)(*gridSize), (int)(*blockSize)>>>(buffer_neighbor_list_pointer,
		buffer_neighbor_list, 
		buffer_delta_V, 
		buffer_x, buffer_y, buffer_z,
		buffer_V_dot_C_array,
		buffer_iter_neighbor_list, 
		buffer_node, buffer_small_delta, buffer_k_bulk_mod);
		
	cudaMemcpy(V_dot_C_array, buffer_V_dot_C_array, (size_t)(*node) *sizeof(float), cudaMemcpyDeviceToHost); 		
		
	float V_dot_C=0.0;
	for (size_t i = 0; i < (size_t)(*node); ++i) {	
		if(V_dot_C_array[i]>V_dot_C){ //find max
			V_dot_C=V_dot_C_array[i];
		}
	}
	
	float delta_t_critical=sqrt(2.0*(float)(*ro)/(V_dot_C));
	
	cudaFree(buffer_V_dot_C_array);
	delete [] V_dot_C_array;
	
	return delta_t_critical;
}

__global__ void u_reinitialization (float *u_dot_x_n0,
	 float *u_dot_y_n0,
	 float *u_dot_z_n0,
	 float *u_doubledot_x_n0,
	 float *u_doubledot_y_n0,
	 float *u_doubledot_z_n0,
	 float *u_dot_x_nhalf,
	 float *u_dot_y_nhalf,
	 float *u_dot_z_nhalf,
	 float *u_x_n0,
	 float *u_y_n0,
	 float *u_z_n0,
	 float *u_x_n1,
	 float *u_y_n1,
	 float *u_z_n1,
	 const float *delta_t){ 
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	u_dot_x_nhalf[i]=u_dot_x_n0[i]+((float)(*delta_t)/2.0*u_doubledot_x_n0[i]);
	u_dot_y_nhalf[i]=u_dot_y_n0[i]+((float)(*delta_t)/2.0*u_doubledot_y_n0[i]);
	u_dot_z_nhalf[i]=u_dot_z_n0[i]+((float)(*delta_t)/2.0*u_doubledot_z_n0[i]);
	
	
	u_x_n1[i]=u_x_n0[i]+((float)(*delta_t)*u_dot_x_nhalf[i]);
	u_y_n1[i]=u_y_n0[i]+((float)(*delta_t)*u_dot_y_nhalf[i]);
	u_z_n1[i]=u_z_n0[i]+((float)(*delta_t)*u_dot_z_nhalf[i]);
		
}

__global__ void cal_displacement(float *f_x, float *f_y, float *f_z,
	 float *b_x, float *b_y, float *b_z,
	 float *u_dot_x_nhalf, float *u_dot_y_nhalf, float *u_dot_z_nhalf,
	 float *u_x_n0, float *u_y_n0, float *u_z_n0,
	 float *u_x_n1, float *u_y_n1, float *u_z_n1,
	 float *u_dot_x_n0, float *u_dot_y_n0, float *u_dot_z_n0,
	 float *u_doubledot_x_n0, float *u_doubledot_y_n0, float *u_doubledot_z_n0, 
	 const float *ro, const float *delta_t){ 
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;

	float u_doubledot_x_n1=(f_x[i]+b_x[i])/(float)(*ro); //no need to use array
	float u_doubledot_y_n1=(f_y[i]+b_y[i])/(float)(*ro);
	float u_doubledot_z_n1=(f_z[i]+b_z[i])/(float)(*ro);
	
	float u_dot_x_n1=u_dot_x_nhalf[i]+((float)(*delta_t)/2.0*u_doubledot_x_n1); //no need to use array
	float u_dot_y_n1=u_dot_y_nhalf[i]+((float)(*delta_t)/2.0*u_doubledot_y_n1);
	float u_dot_z_n1=u_dot_z_nhalf[i]+((float)(*delta_t)/2.0*u_doubledot_z_n1);
	
	//Re-initialization
			
	u_x_n0[i]=u_x_n1[i]; //copy to host
	u_y_n0[i]=u_y_n1[i];  //copy to host
	u_z_n0[i]=u_z_n1[i];  //copy to host
	u_dot_x_n0[i]=u_dot_x_n1;  	//copy to host
	u_dot_y_n0[i]=u_dot_y_n1;	 //copy to host
	u_dot_z_n0[i]=u_dot_z_n1;	 //copy to host
	
	//printf("u_doubledot_x_n1 kernel= %f \n",u_doubledot_x_n1);
	u_doubledot_x_n0[i]=u_doubledot_x_n1;	//copy to host
	u_doubledot_y_n0[i]=u_doubledot_y_n1;	//copy to host
	u_doubledot_z_n0[i]=u_doubledot_z_n1;	//copy to host

}

//This kernel is not needed as it is already integrated in cal_pforce
__global__ void pforce_reinitialization(float *f_x, float *f_y, float *f_z){
		
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	f_x[i]=0.0;
	f_y[i]=0.0;
	f_z[i]=0.0;
}

__global__ void cal_pforce(size_t *neighbor_list_pointer,
	size_t *neighbor_list,
	float *m, 
	float *theta,  float *delta_V,
	float *x,	 float *y,  float *z , 
	float *f_x,	 float *f_y,  float *f_z,
	float *u_x_n1,	 float *u_y_n1,  float *u_z_n1, 
	size_t *iter_neighbor_list,
	const size_t *node,
	const float *small_delta,
	const float *mu ,
	const float *k_bulk_mod	){

	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	f_x[i] =0.0; 	f_y[i] =0.0; 	f_z[i] =0.0;
	
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
				
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
				
		float eta_x=u_x_n1[j]-u_x_n1[i];
		float eta_y=u_y_n1[j]-u_y_n1[i];
		float eta_z=u_z_n1[j]-u_z_n1[i];
				
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		float xi_plus_eta=sqrt(pow((float)(xi_x+eta_x),(float)2.0)\
						+pow((float)(xi_y+eta_y),(float)2.0)\
						+pow((float)(xi_z+eta_z),(float)2.0));
		float e=xi_plus_eta-sqrt(xi_square);	//extension state	
				
		float omega=exp(-xi_square/((float)(*small_delta)*(float)(*small_delta)));
		float f_temp = ((3.0*(float)(*k_bulk_mod) - 5.0*(float)(*mu)) * (theta[i]/m[i]*omega + theta[j]/m[j]*omega)\
				*sqrt(xi_square) + 15.0*(float)(*mu)*(omega/m[i] + omega/m[j])*e) *delta_V[j];
				//mu = G = shear modulus

		f_x[i] = f_x[i]+ (f_temp*(eta_x + xi_x)/ xi_plus_eta );
		f_y[i] = f_y[i]+ (f_temp*(eta_y + xi_y)/ xi_plus_eta );
		f_z[i] = f_z[i]+ (f_temp*(eta_z + xi_z)/ xi_plus_eta );
	}
}

__global__ void weighted_vol(size_t *neighbor_list_pointer,
	size_t *neighbor_list,	float *delta_V, float *m,
	float *x, float *y,  float *z,
	size_t *iter_neighbor_list, const size_t *node,
	const float *small_delta){
	 
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	m[i]=0.0;
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
		
		float xi_x= x[j] - x[i];
		float xi_y= y[j] - y[i];
		float xi_z= z[j] - z[i];		
		
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		
		float omega=exp(-xi_square/((float)(*small_delta)*(float)(*small_delta)));
		m[i] += omega*xi_square*delta_V[j];
		
	}
	
}

__global__ void cal_dilatation(size_t *neighbor_list_pointer,
	 size_t *neighbor_list,
	 float *delta_V,
	 float *theta,
	 float *m,
	 float *x,	 float *y,  float *z,
	 float *u_x_n1,	 float *u_y_n1,  float *u_z_n1,
	 size_t *iter_neighbor_list,
	 const size_t *node,
	 const float *small_delta){
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
		
	theta[i]=0.0;
			
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
		
		float eta_x=u_x_n1[j]-u_x_n1[i];
		float eta_y=u_y_n1[j]-u_y_n1[i];
		float eta_z=u_z_n1[j]-u_z_n1[i];
		
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		float omega=exp(-xi_square/((float)(*small_delta)*(float)(*small_delta)));
		float xi_plus_eta=sqrt(pow((float)(xi_x+eta_x),(float)2.0)\
						+pow((float)(xi_y+eta_y),(float)2.0)\
						+pow((float)(xi_z+eta_z),(float)2.0));
		float e=xi_plus_eta-sqrt(xi_square);	//extension state		
		
		theta[i]=theta[i]+(3.0/m[i]*omega*(sqrt(xi_square))*e*delta_V[j]);
	}
	
}
__global__ void u_reinitialization_BC_dilatation_pforce (float *u_dot_x_n0,
	 float *u_dot_y_n0,
	 float *u_dot_z_n0,
	 float *u_doubledot_x_n0,
	 float *u_doubledot_y_n0,
	 float *u_doubledot_z_n0,
	 float *u_dot_x_nhalf,
	 float *u_dot_y_nhalf,
	 float *u_dot_z_nhalf,
	 float *u_x_n0,
	 float *u_y_n0,
	 float *u_z_n0,
	 float *u_x_n1,
	 float *u_y_n1,
	 float *u_z_n1,
	 const float *delta_t,
	  //for BC
	size_t *ndivy, size_t *ndivz,
	
	//for dilatation calculation :
	
	 size_t *neighbor_list_pointer,
	 size_t *neighbor_list,
	 float *delta_V,
	 float *theta,
	 float *m,
	 float *x,	 float *y,  float *z,
	 size_t *iter_neighbor_list,
	 const size_t *node,
	 const float *small_delta,
	
	//for pforce_reinitialization
	 float *f_x, float *f_y, float *f_z,
	
	//for pforce calculation
	 float *b_x,  float *b_y,  float *b_z,	
	 const float *k_bulk_mod,
	 const float *mu ){
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	
	
	u_dot_x_nhalf[i]=u_dot_x_n0[i]+(float)(*delta_t)/2.0*u_doubledot_x_n0[i];
	u_dot_y_nhalf[i]=u_dot_y_n0[i]+(float)(*delta_t)/2.0*u_doubledot_y_n0[i];
	u_dot_z_nhalf[i]=u_dot_z_n0[i]+(float)(*delta_t)/2.0*u_doubledot_z_n0[i];
	
	__syncthreads();
	
	if(i==10){
		//printf("u_doubledot_x_n0[i]= %f\n", u_doubledot_x_n0[i]);
		
	}
	
	u_x_n1[i]=u_x_n0[i]+((float)(*delta_t)*u_dot_x_nhalf[i]);
	u_y_n1[i]=u_y_n0[i]+((float)(*delta_t)*u_dot_y_nhalf[i]);
	u_z_n1[i]=u_z_n0[i]+((float)(*delta_t)*u_dot_z_nhalf[i]);
	
	
	//Apply BC
	if (i < 4*(size_t)(*ndivy)*(size_t)(*ndivz)) { // in the beginning of the block
			//u_x_n0[i]=0.0; 	u_y_n0[i]=0.0;	u_z_n0[i]=0.0;
			//u_dot_x_n0[i]=0.0; 	u_dot_y_n0[i]=0.0;	u_dot_z_n0[i]=0.0;
			//u_doubledot_x_n0[i]=0.0; u_doubledot_y_n0[i]=0.0; u_doubledot_z_n0[i]=0.0;
			u_dot_x_nhalf[i]=0.0; 	u_dot_y_nhalf[i]=0.0;	u_dot_z_nhalf[i]=0.0;
			u_x_n1[i]=0.0;          u_y_n1[i]=0.0;          u_z_n1[i]=0.0;
	}
	
	__syncthreads();
	
	
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	
	theta[i]=0.0;
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
				
		float eta_x=u_x_n1[j]-u_x_n1[i];
		float eta_y=u_y_n1[j]-u_y_n1[i];
		float eta_z=u_z_n1[j]-u_z_n1[i];
		
		
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		float omega=exp(-xi_square/((float)(*small_delta)*(float)(*small_delta)));
		float xi_plus_eta=sqrt(pow((float)(xi_x+eta_x),(float)2.0)\
						+pow((float)(xi_y+eta_y),(float)2.0)\
						+pow((float)(xi_z+eta_z),(float)2.0));
		float e=xi_plus_eta-sqrt(xi_square);	//extension state			
		
		theta[i]=theta[i]+(3.0/m[i]*omega*(sqrt(xi_square))*e*delta_V[j]);
		
	}
	
	__syncthreads();
	
	//pforce calculation
	
	f_x[i] =0.0; 	f_y[i] =0.0; 	f_z[i] =0.0;
	
	__syncthreads();
	
	//float f_temp2=0.0;
	
	for (size_t k = k_start; k < k_stop; ++k){
		
		size_t j=neighbor_list[k];
				
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
				
		float eta_x=u_x_n1[j]-u_x_n1[i];
		float eta_y=u_y_n1[j]-u_y_n1[i];
		float eta_z=u_z_n1[j]-u_z_n1[i];
				
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		float xi_plus_eta=sqrt(pow((float)(xi_x+eta_x),(float)2.0)\
						+pow((float)(xi_y+eta_y),(float)2.0)\
						+pow((float)(xi_z+eta_z),(float)2.0));
		float e=xi_plus_eta-sqrt(xi_square);	//extension state	
				
		float omega=exp(-xi_square/((float)(*small_delta)*(float)(*small_delta)));
		/*float f_temp = ((3.0*(float)(*k_bulk_mod) - 5.0*(float)(*mu)) * (theta[i]/m[i]*omega + theta[j]/m[j]*omega)\
				*sqrt(xi_square) + 15.0*(float)(*mu)*(omega/m[i] + omega/m[j])*e) *delta_V[j];
				//mu = G = shear modulus
		*/
		float aa = (theta[i]/m[i]*omega + theta[j]/m[j]*omega);
		
		if ( (i==(size_t)(*node)-10) && (j==(size_t)(*node)-3) ){
			
			printf("aa= %f\n", aa);
		}
				
		float f_temp =((3.0f*(float)(*k_bulk_mod) - 5.0f*(float)(*mu))* (theta[i]/m[i]*omega + theta[j]/1.0f*omega )\
			*sqrt(xi_square) + 15.0f*1.0*e ) *delta_V[j]; //dummy value
		
		f_x[i] = f_x[i]+ (f_temp*(eta_x + xi_x)/ xi_plus_eta );
		f_y[i] = f_y[i]+ (f_temp*(eta_y + xi_y)/ xi_plus_eta );
		f_z[i] = f_z[i]+ (f_temp*(eta_z + xi_z)/ xi_plus_eta );
		
		
		__syncthreads();
	}
	
	
	__syncthreads();
	
	
}

/*
//Shared memory
__global__ void cal_dilatation(size_t *neighbor_list_pointer,
	 size_t *neighbor_list,
	 float *delta_V,
	 float *theta,
	 float *m,
	 float *x,	 float *y,  float *z,
	 float *u_x_n1,	 float *u_y_n1,  float *u_z_n1,
	 size_t *iter_neighbor_list,
	 const size_t *node,
	 const float *small_delta){
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
	
	__shared__ size_t node_s;
	__shared__ float small_delta_s;
	
	node_s=(size_t)(*node);
	small_delta_s = (float)(*small_delta);
	
	theta[i]=0.0;
			
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
		
		float eta_x=u_x_n1[j]-u_x_n1[i];
		float eta_y=u_y_n1[j]-u_y_n1[i];
		float eta_z=u_z_n1[j]-u_z_n1[i];
		
		float xi_square=pow((float)xi_x,(float)2.0)\
						+pow((float)xi_y,(float)2.0)\
						+pow((float)xi_z,(float)2.0);
		float omega=exp(-xi_square/(small_delta_s * small_delta_s ));
		float xi_plus_eta=sqrt(pow((float)(xi_x+eta_x),(float)2.0)\
						+pow((float)(xi_y+eta_y),(float)2.0)\
						+pow((float)(xi_z+eta_z),(float)2.0));
		float e=xi_plus_eta-sqrt(xi_square);	//extension state	
		
		theta[i]=theta[i]+(3.0/m[i]*omega*(sqrt(xi_square))*e*delta_V[j]);
	}
}*/

/*
//Explicitly use floating value for math
__global__ void cal_dilatation(size_t *neighbor_list_pointer,
	 size_t *neighbor_list,
	 float *delta_V,
	 float *theta,
	 float *m,
	 float *x,	 float *y,  float *z,
	 float *u_x_n1,	 float *u_y_n1,  float *u_z_n1,
	 size_t *iter_neighbor_list,
	 const size_t *node,
	 const float *small_delta){
	
	size_t i= threadIdx.x+ blockDim.x*threadIdx.y+blockDim.x*blockDim.y*blockIdx.x;
		
	theta[i]=0.0;
			
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop =  0;	
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
	}
	else {
		k_stop =  neighbor_list_pointer[i+1];
	}
	for (size_t k = k_start; k < k_stop; ++k){
		size_t j=neighbor_list[k];
		float xi_x=x[j]-x[i];
		float xi_y=y[j]-y[i];
		float xi_z=z[j]-z[i];
				
		float eta_x=u_x_n1[j]-u_x_n1[i];
		float eta_y=u_y_n1[j]-u_y_n1[i];
		float eta_z=u_z_n1[j]-u_z_n1[i];
		float xi_square=powf((float)xi_x,(float)2.0)\
						+powf((float)xi_y,(float)2.0)\
						+powf((float)xi_z,(float)2.0);
		float omega=expf(-xi_square/((float)(*small_delta)*(float)(*small_delta)));
		float xi_plus_eta=sqrtf(powf((float)(xi_x+eta_x),(float)2.0)\
						+powf((float)(xi_y+eta_y),(float)2.0)\
						+powf((float)(xi_z+eta_z),(float)2.0));
		float e=xi_plus_eta-sqrtf(xi_square);	//extension state			
		
		theta[i]=theta[i]+(3.0/m[i]*omega*(sqrtf(xi_square))*e*delta_V[j]);
	}
}*/