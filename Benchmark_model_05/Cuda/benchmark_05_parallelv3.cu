/*
Serialized all kernels that are slower in GPU
	-cal_displacement,
	-pforce_reinitialization,
	-u_reinitialization 
Use Pinned-Memory in Host

Serial result : 4882 ms (10240 nodes)
Parallel result : - ms
Parallel result with partial serialized: 3492 ms

*/
#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <fstream> // for writing to file
#include <math.h>       //exp, pi
#include <chrono>	//for time measurement
#include <fstream>
#include <ctime>
#include "kernel_main.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	const float length = 1.0; //X
	const float width = 1.0e-1; //Y
	const float heigth = 1.0e-1; //Z
	const float load = 200.0e6; //Newton
	
	const size_t ndivx = 32*4;		//must be a multiply of 32
	//const size_t ndivx = 4;
	//const size_t ndivy = 3;
	const size_t ndivy = 10;
	//const size_t ndivz = 2;
	const size_t ndivz = 8;
	const size_t node = ndivx*ndivy*ndivz;
	
	size_t *neighbor_list_pointer = new size_t[node];
	size_t *neighbor_list = new size_t[node*node/2]; //assume length node/2
	//size_t *neighbor_list = new size_t[node*300];//assume length 200
	
	const float delta=length/ndivx;
	const float small_delta=3.015*delta; //horizon
	const float delta_x=length/ndivx;
	const float delta_y=width/ndivy;
	const float delta_z=heigth/ndivz;
	
	float *x ;	float *y ; float *z ;
	float *delta_V;
	
	cudaMallocHost((void**)&x, node*sizeof(float));
	cudaMallocHost((void**)&y, node*sizeof(float));
	cudaMallocHost((void**)&z, node*sizeof(float));
	cudaMallocHost((void**)&delta_V, node*sizeof(float));
	
	float *m ; // weight
	
	cudaMallocHost((void**)&m, node*sizeof(float));
	
	float *x_plus_ux = new float[node];
	float *y_plus_uy = new float[node];
	float *z_plus_uz = new float[node];
	float *u_n1 = new float[node];
	
	float *f_x;	float *f_y;	float *f_z;
	
	cudaMallocHost((void**)&f_x, node*sizeof(float));
	cudaMallocHost((void**)&f_y, node*sizeof(float));
	cudaMallocHost((void**)&f_z, node*sizeof(float));
	
	float *b_x;	float *b_y;	float *b_z; //body force
	
	cudaMallocHost((void**)&b_x, node*sizeof(float));
	cudaMallocHost((void**)&b_y, node*sizeof(float));
	cudaMallocHost((void**)&b_z, node*sizeof(float));
	
	float *theta;	//dilation	
	
	cudaMallocHost((void**)&theta, node*sizeof(float));
	
	float *u_x_n0 ; float *u_x_n1 ;
	float *u_y_n0 ;	float *u_y_n1 ;
	float *u_z_n0 ; float *u_z_n1 ;	
	
	cudaMallocHost((void**)&u_x_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_y_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_z_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_x_n1, node*sizeof(float));
	cudaMallocHost((void**)&u_y_n1, node*sizeof(float));
	cudaMallocHost((void**)&u_z_n1, node*sizeof(float));
	
	
	float *u_dot_x_n0;	float *u_dot_x_nhalf;
	float *u_dot_y_n0;	float *u_dot_y_nhalf;
	float *u_dot_z_n0;	float *u_dot_z_nhalf;
	
	cudaMallocHost((void**)&u_dot_x_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_dot_y_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_dot_z_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_dot_x_nhalf, node*sizeof(float));
	cudaMallocHost((void**)&u_dot_y_nhalf, node*sizeof(float));
	cudaMallocHost((void**)&u_dot_z_nhalf, node*sizeof(float));
	
	float *u_doubledot_x_n0; 
	float *u_doubledot_y_n0; 
	float *u_doubledot_z_n0; 
	
	cudaMallocHost((void**)&u_doubledot_x_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_doubledot_y_n0, node*sizeof(float));
	cudaMallocHost((void**)&u_doubledot_z_n0, node*sizeof(float));
	
	//------------------------------------------------
	
	cudaMemset(b_x, 0, node*sizeof(float));
	cudaMemset(b_y, 0, node*sizeof(float));
	cudaMemset(b_z, 0, node*sizeof(float));
	
	/*cudaMemset(f_x, 0, node*sizeof(float));
	cudaMemset(f_y, 0, node*sizeof(float));
	cudaMemset(f_z, 0, node*sizeof(float));
	
	cudaMemset(theta, 0, node*sizeof(float));
	
	cudaMemset(u_x_n0, 0, node*sizeof(float));
	cudaMemset(u_y_n0, 0, node*sizeof(float));
	cudaMemset(u_z_n0, 0, node*sizeof(float));
	cudaMemset(u_x_n1, 0, node*sizeof(float));
	cudaMemset(u_y_n1, 0, node*sizeof(float));
	cudaMemset(u_z_n1, 0, node*sizeof(float));
	
	cudaMemset(u_dot_x_n0, 0, node*sizeof(float));
	cudaMemset(u_dot_y_n0, 0, node*sizeof(float));
	cudaMemset(u_dot_z_n0, 0, node*sizeof(float));
	cudaMemset(u_dot_x_nhalf, 0, node*sizeof(float));
	cudaMemset(u_dot_y_nhalf, 0, node*sizeof(float));
	cudaMemset(u_dot_z_nhalf, 0, node*sizeof(float));
	
	cudaMemset(u_doubledot_x_n0, 0, node*sizeof(float));
	cudaMemset(u_doubledot_y_n0, 0, node*sizeof(float));
	cudaMemset(u_doubledot_z_n0, 0, node*sizeof(float));*/
	
	
	const float E = 200.0e9; // Young's modulus
	const float nu=0.25; //Poisson's ratio
	const float mu=E/(2.0*(1.0+nu)); //shear modulus
	const float k_bulk_mod=E/(3.0*(1.0-2.0*nu)); // bulk modulus
	const float ro=7850.0; // mass densiy
	
	size_t iter_neighbor_list_pointer=0; 
	size_t iter_neighbor_list=0; //length of neighbor_list
	
	cout<<"No of nodes = "<<ndivx<<"x"<<ndivy<<"x"<<ndivz<<" = "<<node<<endl;
	
	for (size_t i = (node-4*(ndivy*ndivz)); i < node; ++i) {
		b_x[i] = load/delta_x; //load to end points
	}
	
	//########################################################################################
	//PARALLEL VERSION
	cout<<"Start parallel version with pinned memory"<<endl;
	
	int blockSize;      // The launch configurator returned block size 
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    int gridSize;       // The actual grid size needed, based on input size 
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, discretize_blocks, 0, node); 
	gridSize = (node + blockSize - 1) / blockSize; 
	
	
	printf("\t discretize_blocks Blocksize= %i, ", blockSize);
	printf("minGridSize= %i, ",minGridSize);
	printf("gridSize= %i \n",gridSize);
	
	int blockSize_02; int minGridSize_02;  int gridSize_02;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_02, &blockSize_02, weighted_vol, 0, node);
	gridSize_02 = (node + blockSize_02 - 1) / blockSize_02; 
	
	printf("\t weighted_vol= Blocksize= %i, ", blockSize_02);
	printf("minGridSize= %i, ",minGridSize_02);
	printf("gridSize= %i \n",gridSize_02);
	
	int blockSize_03; int minGridSize_03;  int gridSize_03;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize_03, &blockSize_03, cal_dilatation, 0, node);
	gridSize_03 = (node + blockSize_03 - 1) / blockSize_03;
	
	printf("\t cal_dilatation= Blocksize= %i, ", blockSize_03);
	printf("minGridSize= %i, ",minGridSize_03);
	printf("gridSize= %i \n",gridSize_03);
	
	dim3 gridDim_not_optimized(node/1024,1,1);         // 512 x 1 x 1
	dim3 blockDim_not_optimized(1024, 1,1); // 1024 x 1024 x 1
	
	printf ("Use gridDim = %i, ", gridDim_not_optimized.x);
	printf ("blockDim = %i \n", blockDim_not_optimized.x);
	
	cudaDeviceSynchronize(); //CPU timer synchronization: synchronize CPU thread with GPU
	system_clock::time_point start_parallel = system_clock::now();
	// Create memory buffers on the device for each vector ------------------------
	
	size_t* buffer_neighbor_list_pointer;
	float* buffer_delta_V, * buffer_theta, * buffer_m;
	float* buffer_x, *buffer_y, *buffer_z;
	float* buffer_u_x_n0, * buffer_u_y_n0, * buffer_u_z_n0;
	float* buffer_u_x_n1,* buffer_u_y_n1,* buffer_u_z_n1;
	float* buffer_u_dot_x_n0, *buffer_u_dot_y_n0, *buffer_u_dot_z_n0;
	float* buffer_u_dot_x_nhalf,* buffer_u_dot_y_nhalf, * buffer_u_dot_z_nhalf;
	float* buffer_u_doubledot_x_n0,* buffer_u_doubledot_y_n0,* buffer_u_doubledot_z_n0;
	size_t* buffer_iter_neighbor_list, *buffer_node;
	float* buffer_small_delta, *buffer_delta_t;
	float* buffer_delta_x, * buffer_delta_y, * buffer_delta_z;
	size_t* buffer_ndivx, *buffer_ndivy, *buffer_ndivz;
	
	cudaMalloc((void**)&buffer_neighbor_list_pointer, node*sizeof(size_t));
	cudaMalloc((void**)&buffer_delta_V, node*sizeof(float));
	cudaMalloc((void**)&buffer_theta, node*sizeof(float));
	cudaMalloc((void**)&buffer_m, node*sizeof(float));
	cudaMalloc((void**)&buffer_x, node*sizeof(float));
	cudaMalloc((void**)&buffer_y, node*sizeof(float));
	cudaMalloc((void**)&buffer_z, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_x_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_y_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_z_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_x_n1, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_y_n1, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_z_n1, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_dot_x_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_dot_y_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_dot_z_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_dot_x_nhalf, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_dot_y_nhalf, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_dot_z_nhalf, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_doubledot_x_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_doubledot_y_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_u_doubledot_z_n0, node*sizeof(float));
	cudaMalloc((void**)&buffer_iter_neighbor_list, sizeof(size_t));
	cudaMalloc((void**)&buffer_node, sizeof(size_t));
	cudaMalloc((void**)&buffer_small_delta, sizeof(float));
	cudaMalloc((void**)&buffer_delta_x, sizeof(float));
	cudaMalloc((void**)&buffer_delta_y, sizeof(float));
	cudaMalloc((void**)&buffer_delta_z, sizeof(float));
	cudaMalloc((void**)&buffer_delta_t, sizeof(float));
	cudaMalloc((void**)&buffer_ndivx, sizeof(size_t));
	cudaMalloc((void**)&buffer_ndivy, sizeof(size_t));
	cudaMalloc((void**)&buffer_ndivz, sizeof(size_t));
	
	
	//buffer_neighbor_list is below as iter_neighbor_list is not yet known
			
	//Write buffer for initial values of the problem
	cudaMemcpy(buffer_node, &node, sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_small_delta, &small_delta, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_delta_x, &delta_x, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_delta_y, &delta_y, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_delta_z, &delta_z, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_ndivx, &ndivx, sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_ndivy, &ndivy, sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_ndivz, &ndivz, sizeof(size_t), cudaMemcpyHostToDevice);
	
	
	//Neighbor list search, critical delta_t, x, y, z, delta_V are below
	
	//Discretization

	
	discretize_blocks<<<gridDim_not_optimized, blockDim_not_optimized>>>(buffer_delta_x, buffer_delta_y, buffer_delta_z,
		buffer_x, buffer_y, buffer_z,
		buffer_delta_V, buffer_ndivx, buffer_ndivy, buffer_ndivz);
		
	cudaMemcpy(x, buffer_x, node *sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, buffer_y, node *sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z, buffer_z, node *sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(delta_V, buffer_delta_V, node *sizeof(float), cudaMemcpyDeviceToHost);
	
	//Neighbor list search
	for(size_t i = 0; i < node; ++i){
		neighbor_list_pointer[i]=iter_neighbor_list_pointer;
		for(size_t j = 0; j < node; ++j){
			if (i!=j){
				float distance =sqrt(pow((x[i]-x[j]),2)+pow((y[i]-y[j]),2)+pow((z[i]-z[j]),2));
				if (distance<small_delta){
					neighbor_list[iter_neighbor_list] =j;
					iter_neighbor_list += 1;
					iter_neighbor_list_pointer +=1;
				}
			}
		}//end of j
	}
	cout<<"\t iter_neighbor_list= "<<iter_neighbor_list<<endl;
	
	size_t* buffer_neighbor_list;
	
	cudaMemcpy(buffer_iter_neighbor_list, &iter_neighbor_list, sizeof(size_t), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&buffer_neighbor_list, iter_neighbor_list*sizeof(size_t));

	cudaMemcpy(buffer_neighbor_list_pointer, neighbor_list_pointer, node*sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_neighbor_list, neighbor_list, iter_neighbor_list*sizeof(size_t), cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();
	
	//Critical time step (improvement only 5 % in parallel version)
	
	
	float V_dot_C=0.0;
	float V_dot_C_temp=0.0;
	for (size_t i = 0; i < node; ++i) {	
			V_dot_C_temp=0.0; //Re-initialization
			
			size_t k_start=neighbor_list_pointer[i];
			size_t k_stop=0;
			if(i!=(node-1)){
				k_stop=neighbor_list_pointer[i+1];
			}
			if(i==(node-1)){
				k_stop=iter_neighbor_list;
			}
			for (size_t k = k_start; k < k_stop; ++k){
				size_t j=neighbor_list[k];
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				
				float xi_square=pow(xi_x,2.0)+pow(xi_y,2.0)+pow(xi_z,2.0);
				float C_p=18.0*k_bulk_mod/(sqrt(xi_square)*M_PI*pow(small_delta,4));
				V_dot_C_temp += (C_p*delta_V[j]);
			}
			if(V_dot_C_temp>V_dot_C){ //find max
				V_dot_C=V_dot_C_temp;
			}
	}
	
	//const float delta_t_critical=(length/ndivx)/sqrt(k/ro); //CLF method
	const float delta_t_critical=sqrt(2.0*ro/(V_dot_C));	
	
	cout << "delta_t_critical = "<<delta_t_critical<<endl;
	const float delta_t=delta_t_critical*1.0;	// safety factor = 1.0
	cout << "delta_t = "<<delta_t<<endl;
	
	cudaMemcpy(buffer_delta_t, &delta_t, sizeof(float), cudaMemcpyHostToDevice);
	
	//const float T=(100.0*delta_t);
	const float T=(200.0*delta_t);
	//const float T=(2.0*delta_t);
	const size_t num_steps= T/delta_t;
	
	//###########################################################################
	//Algo I (Linear Peridynamic Solid Initialization)	
	
	weighted_vol<<<gridDim_not_optimized, blockDim_not_optimized>>>( buffer_neighbor_list_pointer,
		buffer_neighbor_list, buffer_delta_V, buffer_m, 
		buffer_x, buffer_y, buffer_z,
		buffer_iter_neighbor_list,  buffer_node, buffer_small_delta);
	
	
	cudaMemcpy(m, buffer_m, node *sizeof(float), cudaMemcpyDeviceToHost);
	
	//Main kernel
	ofstream file_17;
	file_17.open ("disp_cpp.txt");
	
	for (size_t t_step = 0; t_step < num_steps; ++t_step){
		if(t_step%50==0){
			cout<<"Time step t=" <<t_step<< endl;
		}
		//First partial velocity update & nodal displacement (serialized due to slower in GPU)
		
		for (size_t i = 0; i < node; ++i) {
			u_dot_x_nhalf[i]=u_dot_x_n0[i]+(delta_t/2.0*u_doubledot_x_n0[i]);
			u_dot_y_nhalf[i]=u_dot_y_n0[i]+(delta_t/2.0*u_doubledot_y_n0[i]);
			u_dot_z_nhalf[i]=u_dot_z_n0[i]+(delta_t/2.0*u_doubledot_z_n0[i]);
			
			u_x_n1[i]=u_x_n0[i]+(delta_t*u_dot_x_nhalf[i]);
			u_y_n1[i]=u_y_n0[i]+(delta_t*u_dot_y_nhalf[i]);
			u_z_n1[i]=u_z_n0[i]+(delta_t*u_dot_z_nhalf[i]);
		}
		
	
	
		//Apply BC
		for (size_t j = 0; j <4*(ndivy*ndivz); ++j) { // in the beginning of the block
			u_x_n0[j]=0.0; 	u_y_n0[j]=0.0;	u_z_n0[j]=0.0;
			u_dot_x_n0[j]=0.0; 	u_dot_y_n0[j]=0.0;	u_dot_z_n0[j]=0.0;
			u_doubledot_x_n0[j]=0.0; u_doubledot_y_n0[j]=0.0; u_doubledot_z_n0[j]=0.0;
			u_dot_x_nhalf[j]=0.0; 	u_dot_y_nhalf[j]=0.0;	u_dot_z_nhalf[j]=0.0;
			u_x_n1[j]=0.0;          u_y_n1[j]=0.0;          u_z_n1[j]=0.0;
		}
			
		//Compute the dilatation using u at (n+1)
		cudaMemcpy(buffer_u_x_n1, u_x_n1, sizeof(float) * node, cudaMemcpyHostToDevice);
		cudaMemcpy(buffer_u_y_n1, u_y_n1, sizeof(float) * node, cudaMemcpyHostToDevice);
		cudaMemcpy(buffer_u_z_n1, u_z_n1, sizeof(float) * node, cudaMemcpyHostToDevice);
			
		cal_dilatation<<<gridDim_not_optimized, blockDim_not_optimized>>>( buffer_neighbor_list_pointer,
			buffer_neighbor_list, buffer_delta_V, 
			buffer_theta, buffer_m,
			buffer_x, buffer_y, buffer_z,
			buffer_u_x_n1, buffer_u_y_n1, buffer_u_z_n1,
			buffer_iter_neighbor_list, buffer_node,
			buffer_small_delta);
		cudaMemcpy(theta, buffer_theta, node *sizeof(float), cudaMemcpyDeviceToHost);
		
		//Re-initialization peridynamics force (serialized due to slower in GPU)
		for (size_t i = 0; i < node; ++i) {
			f_x[i]=0.0; 	f_y[i]=0.0; 	f_z[i]=0.0;
		}
		
		//Compute the pairwise contributions to the global force density vector
		for (size_t i = 0; i < node; ++i) {
			size_t k_start=neighbor_list_pointer[i];
			size_t k_stop=0;
			if(i!=(node-1)){
				k_stop=neighbor_list_pointer[i+1];
			}
			if(i==(node-1)){
				k_stop=iter_neighbor_list;
			}
			for (size_t k = k_start; k < k_stop; ++k){
				size_t j=neighbor_list[k];
				
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				
				float eta_x=u_x_n1[j]-u_x_n1[i];
				float eta_y=u_y_n1[j]-u_y_n1[i];
				float eta_z=u_z_n1[j]-u_z_n1[i];
				
				float xi_square=pow(xi_x,2)+pow(xi_y,2)+pow(xi_z,2);
				float omega=exp(-xi_square/(small_delta*small_delta));
				float xi_plus_eta=sqrt(pow((xi_x+eta_x),2)+pow((xi_y+eta_y),2)+pow((xi_z+eta_z),2));
				float e=xi_plus_eta-sqrt(xi_square);	//extension state			
				
				
				float e_d=e-(theta[i]*sqrt(xi_square)/3.0);	//deviatoric extension state
				float t=(3.0/m[i]*k_bulk_mod*theta[i]*omega*sqrt(xi_square))+(15.0*mu/m[i]*omega*e_d);
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
		
		//Calculate displacement (serialized due to slower in GPU)
		for (size_t i = 0; i < node; ++i) {
			float u_doubledot_x_n1=(f_x[i]+b_x[i])/ro; //no need to use array
			float u_doubledot_y_n1=(f_y[i]+b_y[i])/ro;
			float u_doubledot_z_n1=(f_z[i]+b_z[i])/ro;
			
			float u_dot_x_n1=u_dot_x_nhalf[i]+(delta_t/2.0*u_doubledot_x_n1); //no need to use array
			float u_dot_y_n1=u_dot_y_nhalf[i]+(delta_t/2.0*u_doubledot_y_n1);
			float u_dot_z_n1=u_dot_z_nhalf[i]+(delta_t/2.0*u_doubledot_z_n1);
			
			//Re-initialization
			
			u_x_n0[i]=u_x_n1[i];
			u_y_n0[i]=u_y_n1[i];
			u_z_n0[i]=u_z_n1[i];
			u_dot_x_n0[i]=u_dot_x_n1;
			u_dot_y_n0[i]=u_dot_y_n1;
			u_dot_z_n0[i]=u_dot_z_n1;
			u_doubledot_x_n0[i]=u_doubledot_x_n1; 
			u_doubledot_y_n0[i]=u_doubledot_y_n1;
			u_doubledot_z_n0[i]=u_doubledot_z_n1;
			
		}
		
		file_17 <<t_step<<"   "<<u_x_n1[(node/2)]<<"   "<<u_x_n1[node-1]<<endl; //disp at end of rope
		
	} //end of time integration
	file_17.close();
	
	cudaDeviceSynchronize(); //CPU timer synchronization: synchronize CPU thread with GPU
	system_clock::time_point stop_parallel = system_clock::now();
	std::chrono::duration<float, std::milli> duration_parallel = stop_parallel - start_parallel;
	cout << "Parallel peridynamics = "<<duration_parallel.count()<<" millisecond"<<endl;
	
	
	for (size_t i = 0; i < node; ++i) {
		u_n1[i]=sqrt(pow(u_x_n1[i],2.0)+pow(u_y_n1[i],2.0)+pow(u_z_n1[i],2.0));
		x_plus_ux[i]=x[i]+u_x_n1[i];
		y_plus_uy[i]=y[i]+u_y_n1[i];
		z_plus_uz[i]=z[i]+u_z_n1[i];
	}
	
	
	float u_n1_sum=0.0;
	for (size_t i = 0; i < node; ++i) {
		u_n1_sum += u_n1[i];
	}
	cout<<"u_n1_sum at the end of time step = "<< u_n1_sum<<endl;
	
	
	ofstream file_18;
	file_18.open ("pos_vs_disp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_18 <<x[i]<<"   "<<u_x_n1[i]<<"   "<<u_y_n1[i]<<"   "<<u_z_n1[i]<<"   "<<u_n1[i];
		if(i < (node - 1)) {
			file_18 <<endl;
		}
		
	}
	file_18.close();
	
	cudaFree(buffer_neighbor_list);
	cudaFree(buffer_neighbor_list_pointer);
	cudaFree(buffer_delta_V); cudaFree(buffer_theta); cudaFree(buffer_m);
	cudaFree(buffer_x); cudaFree(buffer_y); cudaFree(buffer_z);
	cudaFree(buffer_u_x_n0); cudaFree(buffer_u_y_n0); cudaFree(buffer_u_z_n0);
	cudaFree(buffer_u_x_n1); cudaFree(buffer_u_y_n1); cudaFree(buffer_u_z_n1);
	cudaFree(buffer_u_dot_x_n0); cudaFree(buffer_u_dot_y_n0); cudaFree(buffer_u_dot_z_n0);
	cudaFree(buffer_u_dot_x_nhalf); cudaFree(buffer_u_dot_y_nhalf); cudaFree(buffer_u_dot_z_nhalf);
	cudaFree(buffer_u_doubledot_x_n0); cudaFree(buffer_u_doubledot_y_n0); cudaFree(buffer_u_doubledot_z_n0);
	cudaFree(buffer_iter_neighbor_list); cudaFree(buffer_node); 
	cudaFree(buffer_small_delta); cudaFree(buffer_delta_t);
	cudaFree(buffer_delta_x); cudaFree(buffer_delta_y);  cudaFree(buffer_delta_z);
	cudaFree(buffer_ndivx); cudaFree(buffer_ndivy);  cudaFree(buffer_ndivz);
	
	
	delete [] neighbor_list_pointer;
	delete [] neighbor_list;
	delete [] x_plus_ux;	delete [] y_plus_uy; 	delete [] z_plus_uz;
	delete [] u_n1;
	
	cudaFreeHost(x); cudaFreeHost(y); cudaFreeHost(z);
	cudaFreeHost(delta_V);
	cudaFreeHost(m); cudaFreeHost(theta);
	cudaFreeHost(f_x); cudaFreeHost(f_y); cudaFreeHost(f_z);
	cudaFreeHost(b_x); cudaFreeHost(b_y); cudaFreeHost(b_z);
	cudaFreeHost(u_x_n0); cudaFreeHost(u_y_n0); cudaFreeHost(u_z_n0);
	cudaFreeHost(u_x_n1); cudaFreeHost(u_y_n1); cudaFreeHost(u_z_n1);
	cudaFreeHost(u_dot_x_n0); cudaFreeHost(u_dot_y_n0); cudaFreeHost(u_dot_z_n0);
	cudaFreeHost(u_dot_x_nhalf); cudaFreeHost(u_dot_y_nhalf); cudaFreeHost(u_dot_z_nhalf);
	cudaFreeHost(u_doubledot_x_n0); cudaFreeHost(u_doubledot_y_n0); cudaFreeHost(u_doubledot_z_n0);
	
	printf("End of program!");
	
	
}