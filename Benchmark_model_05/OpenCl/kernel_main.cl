//OpenCl kernel for Benchmark 05

/*
//Version1 : Parallelized i & j
__kernel void neighbor_list_search(__global float *neighbor_list,
		__global float *neighbor_list_posize_ter,
		__global float *x, 
		__global float *y, 
		__global float *z,
		__constant float *small_delta,
		__global size_t *iter_neighbor_list){ 
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t iter_neighbor_list_posize_ter=0;	//initialize here instead of in the host?
	
	neighbor_list_posize_ter[i]=iter_neighbor_list_posize_ter;
	
	size_t iter_neighbor_list_val= *iter_neighbor_list;
	float small_delta_val= *small_delta;
	
	if (i!=j){
			float distance =sqrt(pow((x[i]-x[j]),2)+pow((y[i]-y[j]),2)+pow((z[i]-z[j]),2));
			if (distance<small_delta_val){
				neighbor_list[iter_neighbor_list_val]=j;
				iter_neighbor_list_val += 1;
				iter_neighbor_list_posize_ter +=1;
			}
	}
	
}*/

/*
//Version2 : Parallelized i, while j executed within kernel
__kernel void neighbor_list_search(__global float *neighbor_list,
		__global float *neighbor_list_posize_ter,
		__global float *x, 
		__global float *y, 
		__global float *z,
		__constant float *small_delta,
		__global size_t *iter_neighbor_list){ 
	size_t i = get_global_id(0);
	
	size_t iter_neighbor_list_posize_ter=0;	//initialize here instead of in the host?
	
	neighbor_list_posize_ter[i]=iter_neighbor_list_posize_ter;
	
	size_t iter_neighbor_list_val= *iter_neighbor_list;
	float small_delta_val= *small_delta;
	
	size_t node =10 ; //need to be updated
	for(size_t j = 0; j < node; ++j){
		if (i!=j){
				float distance =sqrt(pow((x[i]-x[j]),2)+pow((y[i]-y[j]),2)+pow((z[i]-z[j]),2));
				if (distance<small_delta_val){
					neighbor_list[iter_neighbor_list_val]=j;
					iter_neighbor_list_val += 1;
					iter_neighbor_list_posize_ter_val +=1;
				}
		}
	}
}

//Version3 : Parallelized j
__kernel void neighbor_list_search(__global float *neighbor_list,
		__global float *neighbor_list_posize_ter,
		__global const float *x, 
		__global const float *y, 
		__global const float *z,
		__constant float *small_delta,
		__global size_t *iter_neighbor_list,
		__global size_t *iter_neighbor_list_posize_ter,
		__global size_t *i){ 
	
	size_t j= get_global_id(0);
	float small_delta_val= *small_delta;
	size_t iter_neighbor_list_val= *iter_neighbor_list;
	
	size_t iter_neighbor_list_posize_ter_val= *iter_neighbor_list_posize_ter;
	
	prsize_tf("get_global_size(0): %i \n",get_global_size(0));
	prsize_tf("get_global_size(1): %i \n",get_global_size(1));
	prsize_tf("get_local_size(0): %i \n",get_local_size(0));
	prsize_tf("i= %i ,j= %i \n",*i, j);
	prsize_tf( "x[*i]= %f, x[j]= %f, j= %i \n", x[*i], x[j],j);
	if ((*i)!=j){
		float distance =sqrt(pow((x[*i]-x[j]),2)+pow((y[*i]-y[j]),2)+pow((z[*i]-z[j]),2));
		if (distance<small_delta_val){
			neighbor_list[iter_neighbor_list_val]=j;
			iter_neighbor_list_val += 1;
			iter_neighbor_list_posize_ter_val +=1;
			//prsize_tf("Hit on j= %i , distance= %f \n",j, distance);
		}
	}
	
}
*/
__kernel void discretize_blocks(__global const float *delta_x,
	__global const float *delta_y,
	__global const float *delta_z,
	__global float *x, __global float *y, __global float *z, __global float *delta_V,
	__global const size_t *ndivx,
	__global const size_t *ndivy,
	__global const size_t *ndivz){ 
	
	size_t i= get_global_id(0);

	x[i]=(2.0*floor((float)i/(float)(*ndivy)/(float)(*ndivz)) +1.0)  *(float)(*delta_x)/2.0;
	y[i]=(2*(floor((float)i/ (float)(*ndivz)) - (float)(*ndivy)*floor((float)i/(float)(*ndivy)/(float)(*ndivz))) + 1.0) \
		* (float)(*delta_y)/2.0;
	z[i]= (2.0*(i%(size_t)(*ndivz)) +1.0) * (float)(*delta_z)/2.0;
	delta_V[i]=(float)(*delta_x)*(float)(*delta_y)*(float)(*delta_z);
	
}
__kernel void critical_time_step(__global size_t *neighbor_list_pointer,
	__global size_t *neighbor_list,
	__global float *delta_V,
	__global float *x,	__global float *y, __global float *z, /*6*/
	__global float *V_dot_C_array,
	__global size_t *iter_neighbor_list,
	__global const size_t *node,
	__global const float *small_delta,
	__global const float *k_bulk_mod ){

	size_t i= get_global_id(0);
	
	//__local float V_dot_C_array_device[(int)(*node)]; //variable length is not supported
	//__local float V_dot_C_array_device[30000];
	
	float V_dot_C_temp=0.0; //Re-initialization
			
	size_t k_start=neighbor_list_pointer[i];
	
	size_t k_stop=0;
		if(i!=((size_t)(*node)-1)){
		k_stop=(size_t)neighbor_list_pointer[i+1];
	}
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
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
__kernel void u_reinitialization(__global float *u_dot_x_n0,
	__global float *u_dot_y_n0,
	__global float *u_dot_z_n0,
	__global float *u_doubledot_x_n0,
	__global float *u_doubledot_y_n0,
	__global float *u_doubledot_z_n0,
	__global float *u_dot_x_nhalf,
	__global float *u_dot_y_nhalf,
	__global float *u_dot_z_nhalf,
	__global float *u_x_n0,
	__global float *u_y_n0,
	__global float *u_z_n0,
	__global float *u_x_n1,
	__global float *u_y_n1,
	__global float *u_z_n1,
	__global const float *delta_t){ 
	
	size_t i= get_global_id(0);
	
	u_dot_x_nhalf[i]=u_dot_x_n0[i]+((float)(*delta_t)/2.0*u_doubledot_x_n0[i]);
	u_dot_y_nhalf[i]=u_dot_y_n0[i]+((float)(*delta_t)/2.0*u_doubledot_y_n0[i]);
	u_dot_z_nhalf[i]=u_dot_z_n0[i]+((float)(*delta_t)/2.0*u_doubledot_z_n0[i]);
	
	
	u_x_n1[i]=u_x_n0[i]+((float)(*delta_t)*u_dot_x_nhalf[i]);
	u_y_n1[i]=u_y_n0[i]+((float)(*delta_t)*u_dot_y_nhalf[i]);
	u_z_n1[i]=u_z_n0[i]+((float)(*delta_t)*u_dot_z_nhalf[i]);
		
		
}
__kernel void cal_displacement(__global float *f_x,	__global float *f_y, __global float *f_z,
	__global float *b_x, __global float *b_y, __global float *b_z,
	__global float *u_dot_x_nhalf, __global float *u_dot_y_nhalf, __global float *u_dot_z_nhalf,
	__global float *u_x_n0, __global float *u_y_n0, __global float *u_z_n0,
	__global float *u_x_n1, __global float *u_y_n1, __global float *u_z_n1,
	__global float *u_dot_x_n0, __global float *u_dot_y_n0, __global float *u_dot_z_n0,
	__global float *u_doubledot_x_n0, __global float *u_doubledot_y_n0, __global float *u_doubledot_z_n0, 
	__global const float *ro,
	__global const float *delta_t){ 
	
	int i= get_global_id(0);
	
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
	u_doubledot_x_n0[i]=u_doubledot_x_n1;	//copy to host
	u_doubledot_y_n0[i]=u_doubledot_y_n1;	//copy to host
	u_doubledot_z_n0[i]=u_doubledot_z_n1;	//copy to host
	
}

__kernel void pforce_reinitialization(__global float *f_x,
	__global float *f_y,
	__global float *f_z){
		
	size_t i= get_global_id(0);
	
	f_x[i]=0.0;
	f_y[i]=0.0;
	f_z[i]=0.0;
}


__kernel void weighted_vol(__global size_t *neighbor_list_pointer,
	__global size_t *neighbor_list,
	__global float *delta_V,
	__global float *m,
	__global float *x,	__global float *y, __global float *z,
	__global size_t *iter_neighbor_list,
	__global const size_t *node,
	__global const float *small_delta){
		
	size_t i= get_global_id(0);
	
	m[i]=0.0;
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop=0;
	
	if(i!=((size_t)(*node)-1)){
		k_stop=(size_t)neighbor_list_pointer[i+1];
	}
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
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

__kernel void cal_dilatation(__global size_t *neighbor_list_pointer,
	__global size_t *neighbor_list,
	__global float *delta_V,
	__global float *theta,
	__global float *m,
	__global float *x,	__global float *y, __global float *z,
	__global float *u_x_n1,	__global float *u_y_n1, __global float *u_z_n1,
	__global size_t *iter_neighbor_list,
	__global const size_t *node,
	__global const float *small_delta){
	
	size_t i= get_global_id(0);
		
	theta[i]=0.0;
			
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop=0;
	if(i!=((size_t)(*node)-1)){
		k_stop=neighbor_list_pointer[i+1];
	}
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
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

__kernel void cal_pforce(__global size_t *neighbor_list_pointer,
	__global size_t *neighbor_list,
	__global float *m, 
	__global float *theta, __global float *delta_V,
	__global float *x,	__global float *y, __global float *z , 
	__global float *f_x,	__global float *f_y, __global float *f_z,
	__global float *u_x_n1,	__global float *u_y_n1, __global float *u_z_n1, 
	__global size_t *iter_neighbor_list,
	__global const size_t *node,
	__global const float *small_delta,
	__global const float *mu ,
	__global const float *k_bulk_mod	){

	size_t i= get_global_id(0);
	
	
	size_t k_start=neighbor_list_pointer[i];
	size_t k_stop=0;
	if(i!=((size_t)(*node)-1)){
		k_stop=neighbor_list_pointer[i+1];
	}
	if(i==((size_t)(*node)-1)){
		k_stop=(size_t)(*iter_neighbor_list);
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
				
		float e_d=e-(theta[i]*sqrt(xi_square)/3.0);	//deviatoric extension state
		float t=(3.0/m[i]*(float)(*k_bulk_mod)*theta[i]*omega*sqrt(xi_square))\
			+(15.0*(float)(*mu)/m[i]*omega*e_d);
		float M_x=(xi_x+eta_x)/xi_plus_eta;
		float M_y=(xi_y+eta_y)/xi_plus_eta;
		float M_z=(xi_z+eta_z)/xi_plus_eta;
		
		f_x[i] = f_x[i]+(t*M_x*delta_V[j]);
		f_y[i] = f_y[i]+(t*M_y*delta_V[j]);
		f_z[i] = f_z[i]+(t*M_z*delta_V[j]);
		
		f_x[j] = f_x[j]-(t*M_x*delta_V[i]);
		f_y[j] = f_y[j]-(t*M_y*delta_V[i]);
		f_z[j] = f_z[j]-(t*M_z*delta_V[i]);
	}
}
