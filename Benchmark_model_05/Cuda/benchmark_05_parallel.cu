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
	
	const size_t ndivx = 100;
	//const size_t ndivx = 13;
	//const size_t ndivy = 30;
	const size_t ndivy = 12;
	//const size_t ndivz = 10;
	const size_t ndivz = 12;
	const size_t node = ndivx*ndivy*ndivz;
	
	vector<size_t>neighbor_list_pointer(node,0);
	vector<size_t>neighbor_list(node*node/2,0);//assume length node/2
	//vector<size_t>neighbor_list(node*400,0);//assume length 200
	
	const float delta=length/ndivx;
	const float small_delta=3.015*delta; //horizon
	const float delta_x=length/ndivx;
	const float delta_y=width/ndivy;
	const float delta_z=heigth/ndivz;
	
	vector<float> x(node,0.0);	
	vector<float> y(node,0.0);	
	vector<float> z(node,0.0);
	vector<float> delta_V(node,0.0);
	
	vector<float> m(node,0.0); // weight
	vector<float> x_plus_ux(node,0.0);
	vector<float> y_plus_uy(node,0.0);
	vector<float> z_plus_uz(node,0.0);
	
	vector<float> f_x(node,0.0);
	vector<float> f_y(node,0.0);
	vector<float> f_z(node,0.0);
	
	vector<float> b_x(node,0.0); //body force
	vector<float> b_y(node,0.0);
	vector<float> b_z(node,0.0);
	
	vector< vector<float>> d(node, vector<float>(node,0.0)); //damage variable
	vector<float> theta(node,0.0);	//dilation	
	
	vector<float> epsilon(node,0.0);	
	vector<float> M(node,0.0);
	
	vector<float> u_x_n0(node,0.0);	vector<float> u_x_n1(node,0.0);
	vector<float> u_y_n0(node,0.0);	vector<float> u_y_n1(node,0.0); 	
	vector<float> u_z_n0(node,0.0);	vector<float> u_z_n1(node,0.0); 
	vector<float> u_n1(node,0.0); //scalar of total disp
	vector<float> u_dot_x_n0(node,0.0); vector<float> u_dot_x_nhalf(node,0.0); vector<float> u_dot_x_n1(node,0.0); 	
	vector<float> u_dot_y_n0(node,0.0); vector<float> u_dot_y_nhalf(node,0.0);vector<float> u_dot_y_n1(node,0.0); 
	vector<float> u_dot_z_n0(node,0.0);vector<float> u_dot_z_nhalf(node,0.0);vector<float> u_dot_z_n1(node,0.0);
	vector<float> u_doubledot_x_n0(node,0.0); vector<float> u_doubledot_x_n1(node,0.0); 	
	vector<float> u_doubledot_y_n0(node,0.0); vector<float> u_doubledot_y_n1(node,0.0); 
	vector<float> u_doubledot_z_n0(node,0.0);vector<float> u_doubledot_z_n1(node,0.0);
	
	const float E = 200.0e9; // Young's modulus
	const float nu=0.25; //Poisson's ratio
	const float mu=E/(2.0*(1.0+nu)); //shear modulus
	const float k_bulk_mod=E/(3.0*(1.0-2.0*nu)); // bulk modulus
	const float ro=7850.0; // mass densiy
	
	size_t iter_neighbor_list_pointer=0; 
	size_t iter_neighbor_list=0; //length of neighbor_list
	
	cout<<"No of nodes = "<<node<<endl;
	
	for (size_t i = (node-4*(ndivy*ndivz)); i < node; ++i) {
		b_x[i] = load/delta_x; //load to end points
	}
	
	//########################################################################################
	//PARALLEL VERSION
	cout<<"Start parallel version"<<endl;
	
	
	
	
	
	system_clock::time_point start_parallel = system_clock::now();
		

	// Create memory buffers on the device for each vector 
	cl::Buffer buffer_neighbor_list_pointer(context, CL_MEM_READ_WRITE, sizeof(size_t)*node);
	cl::Buffer buffer_delta_V(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_theta(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_m(context, CL_MEM_READ_WRITE, sizeof(float)*node); 
	cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_y(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_z(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_x_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_y_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_z_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_x_n1(context, CL_MEM_READ_WRITE, sizeof(float)*node); 
	cl::Buffer buffer_u_y_n1(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_z_n1(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_dot_x_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_dot_y_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_dot_z_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_dot_x_nhalf(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_dot_y_nhalf(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_dot_z_nhalf(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_doubledot_x_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_doubledot_y_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_u_doubledot_z_n0(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_f_x(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_f_y(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_f_z(context, CL_MEM_READ_WRITE, sizeof(float)*node);
	cl::Buffer buffer_b_x(context, CL_MEM_READ_ONLY, sizeof(float)*node);
	cl::Buffer buffer_b_y(context, CL_MEM_READ_ONLY, sizeof(float)*node);
	cl::Buffer buffer_b_z(context, CL_MEM_READ_ONLY, sizeof(float)*node);		
	cl::Buffer buffer_iter_neighbor_list(context, CL_MEM_READ_ONLY, sizeof(size_t));
	cl::Buffer buffer_node(context, CL_MEM_READ_ONLY, sizeof(size_t)); 
	cl::Buffer buffer_small_delta(context, CL_MEM_READ_ONLY, sizeof(float));
	cl::Buffer buffer_delta_x(context, CL_MEM_READ_ONLY, sizeof(float));
	cl::Buffer buffer_delta_y(context, CL_MEM_READ_ONLY, sizeof(float));
	cl::Buffer buffer_delta_z(context, CL_MEM_READ_ONLY, sizeof(float));
	cl::Buffer buffer_delta_t(context, CL_MEM_READ_ONLY, sizeof(float));
	cl::Buffer buffer_ro(context, CL_MEM_READ_ONLY, sizeof(float));
	cl::Buffer buffer_ndivx(context, CL_MEM_READ_ONLY, sizeof(size_t));
	cl::Buffer buffer_ndivy(context, CL_MEM_READ_ONLY, sizeof(size_t));
	cl::Buffer buffer_ndivz(context, CL_MEM_READ_ONLY, sizeof(size_t));
	//buffer_neighbor_list is below as iter_neighbor_list is not yet known
	
	cl::CommandQueue queue(context, selectedDevice);
	
	cl::NDRange globalRange(node, 1); //(npoints, nGrids)
	cl::NDRange localRange(1, 1); //(npoints, 1)
	
	//Write buffer for initial values of the problem
	queue.enqueueWriteBuffer(buffer_node, CL_TRUE, 0, sizeof(size_t), &node);
	queue.enqueueWriteBuffer(buffer_small_delta, CL_TRUE, 0, sizeof(float), &small_delta);
	queue.enqueueWriteBuffer(buffer_delta_x, CL_TRUE, 0, sizeof(float), &delta_x);
	queue.enqueueWriteBuffer(buffer_delta_y, CL_TRUE, 0, sizeof(float), &delta_y);
	queue.enqueueWriteBuffer(buffer_delta_z, CL_TRUE, 0, sizeof(float), &delta_z);
	queue.enqueueWriteBuffer(buffer_ndivx, CL_TRUE, 0, sizeof(size_t), &ndivx);
	queue.enqueueWriteBuffer(buffer_ndivy, CL_TRUE, 0, sizeof(size_t), &ndivy);
	queue.enqueueWriteBuffer(buffer_ndivz, CL_TRUE, 0, sizeof(size_t), &ndivz);
	queue.enqueueWriteBuffer(buffer_ro, CL_TRUE, 0, sizeof(float), &ro);
	queue.enqueueWriteBuffer(buffer_b_x, CL_TRUE, 0, sizeof(float) * node, &b_x[0]);
	queue.enqueueWriteBuffer(buffer_b_y, CL_TRUE, 0, sizeof(float) * node, &b_y[0]);
	queue.enqueueWriteBuffer(buffer_b_z, CL_TRUE, 0, sizeof(float) * node, &b_z[0]);
		
	//Neighbor list search, critical delta_t, x, y, z, delta_V are below
	
		
	
	
	//Discretization
	cl::Kernel cl_kernel_discretize_blocks(clProgram, "discretize_blocks");
		
	cl_kernel_discretize_blocks.setArg(0, buffer_delta_x);
	cl_kernel_discretize_blocks.setArg(1, buffer_delta_y);
	cl_kernel_discretize_blocks.setArg(2, buffer_delta_z);
	cl_kernel_discretize_blocks.setArg(3, buffer_x);
	cl_kernel_discretize_blocks.setArg(4, buffer_y);
	cl_kernel_discretize_blocks.setArg(5, buffer_z);
	cl_kernel_discretize_blocks.setArg(6, buffer_delta_V);
	cl_kernel_discretize_blocks.setArg(7, buffer_ndivx);
	cl_kernel_discretize_blocks.setArg(8, buffer_ndivy);
	cl_kernel_discretize_blocks.setArg(9, buffer_ndivz);
	
	cl::Event evt01;
	queue.enqueueNDRangeKernel(cl_kernel_discretize_blocks, cl::NullRange, globalRange, cl::NullRange,	NULL, &evt01);
		
	queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * node, &x[0]);	
	queue.enqueueReadBuffer(buffer_y, CL_TRUE, 0, sizeof(float) * node, &y[0]);	
	queue.enqueueReadBuffer(buffer_z, CL_TRUE, 0, sizeof(float) * node, &z[0]);	
	queue.enqueueReadBuffer(buffer_delta_V, CL_TRUE, 0, sizeof(float) * node, &delta_V[0]);	
	
	queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * node, &x[0]);
	queue.enqueueWriteBuffer(buffer_y, CL_TRUE, 0, sizeof(float) * node, &y[0]);
	queue.enqueueWriteBuffer(buffer_z, CL_TRUE, 0, sizeof(float) * node, &z[0]);
	queue.enqueueWriteBuffer(buffer_delta_V, CL_TRUE, 0, sizeof(float) * node, &delta_V[0]);
	
	
	queue.finish();
	
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
	cout<<"iter_neighbor_list= "<<iter_neighbor_list<<endl;
	cl::Buffer buffer_neighbor_list(context, CL_MEM_READ_WRITE, sizeof(size_t)*iter_neighbor_list);
	queue.enqueueWriteBuffer(buffer_iter_neighbor_list, CL_TRUE, 0, sizeof(size_t), &iter_neighbor_list);
	queue.enqueueWriteBuffer(buffer_neighbor_list_pointer, CL_TRUE, 0, sizeof(size_t) * node, &neighbor_list_pointer[0]);
	queue.enqueueWriteBuffer(buffer_neighbor_list, CL_TRUE, 0, sizeof(size_t) *iter_neighbor_list, &neighbor_list[0]);
	
	
	//Critical time step
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
	
	queue.enqueueWriteBuffer(buffer_delta_t, CL_TRUE, 0, sizeof(float), &delta_t);
	
	//const float T=(100.0*delta_t);
	const float T=(200.0*delta_t);
	//const float T=(2.0*delta_t);
	const float num_steps= T/delta_t;
		
		
	
	//###########################################################################
	//Algo I (Linear Peridynamic Solid Initialization)

	cl::Kernel cl_kernel_weighted_vol(clProgram, "weighted_vol");
	
	cl_kernel_weighted_vol.setArg(0, buffer_neighbor_list_pointer);
	cl_kernel_weighted_vol.setArg(1, buffer_neighbor_list);
	cl_kernel_weighted_vol.setArg(2, buffer_delta_V);
	cl_kernel_weighted_vol.setArg(3, buffer_m);
	cl_kernel_weighted_vol.setArg(4, buffer_x);
	cl_kernel_weighted_vol.setArg(5, buffer_y);
	cl_kernel_weighted_vol.setArg(6, buffer_z);
	cl_kernel_weighted_vol.setArg(7, buffer_iter_neighbor_list);
	cl_kernel_weighted_vol.setArg(8, buffer_node);
	cl_kernel_weighted_vol.setArg(9, buffer_small_delta);
	
	cl::Event evt02;
	queue.enqueueNDRangeKernel(cl_kernel_weighted_vol, cl::NullRange, globalRange, cl::NullRange, NULL, &evt02);
	
	queue.enqueueReadBuffer(buffer_m, CL_TRUE, 0, sizeof(float) * node, &m[0]);
				
	queue.finish();
		
	
	//Creating all kernels for time integration loop
	
	cl::Kernel cl_kernel_pforce_reinitialization(clProgram, "pforce_reinitialization");

	cl_kernel_pforce_reinitialization.setArg(0, buffer_f_x);
	cl_kernel_pforce_reinitialization.setArg(1, buffer_f_y);
	cl_kernel_pforce_reinitialization.setArg(2, buffer_f_z);
	
	cl::Kernel cl_kernel_u_reinitialization(clProgram, "u_reinitialization");
		
	cl_kernel_u_reinitialization.setArg(0, buffer_u_dot_x_n0);
	cl_kernel_u_reinitialization.setArg(1, buffer_u_dot_y_n0);
	cl_kernel_u_reinitialization.setArg(2, buffer_u_dot_z_n0);
	cl_kernel_u_reinitialization.setArg(3, buffer_u_doubledot_x_n0);
	cl_kernel_u_reinitialization.setArg(4, buffer_u_doubledot_y_n0);
	cl_kernel_u_reinitialization.setArg(5, buffer_u_doubledot_z_n0);
	cl_kernel_u_reinitialization.setArg(6, buffer_u_dot_x_nhalf);
	cl_kernel_u_reinitialization.setArg(7, buffer_u_dot_y_nhalf);
	cl_kernel_u_reinitialization.setArg(8, buffer_u_dot_z_nhalf);
	cl_kernel_u_reinitialization.setArg(9, buffer_u_x_n0);
	cl_kernel_u_reinitialization.setArg(10, buffer_u_y_n0);
	cl_kernel_u_reinitialization.setArg(11, buffer_u_z_n0);
	cl_kernel_u_reinitialization.setArg(12, buffer_u_x_n1);
	cl_kernel_u_reinitialization.setArg(13, buffer_u_y_n1);
	cl_kernel_u_reinitialization.setArg(14, buffer_u_z_n1);
	cl_kernel_u_reinitialization.setArg(15, buffer_delta_t);
	
	cl::Kernel cl_kernel_cal_dilatation(clProgram, "cal_dilatation");

	cl_kernel_cal_dilatation.setArg(0, buffer_neighbor_list_pointer);
	cl_kernel_cal_dilatation.setArg(1, buffer_neighbor_list);
	cl_kernel_cal_dilatation.setArg(2, buffer_delta_V);
	cl_kernel_cal_dilatation.setArg(3, buffer_theta);
	cl_kernel_cal_dilatation.setArg(4, buffer_m);
	cl_kernel_cal_dilatation.setArg(5, buffer_x);
	cl_kernel_cal_dilatation.setArg(6, buffer_y);
	cl_kernel_cal_dilatation.setArg(7, buffer_z);
	cl_kernel_cal_dilatation.setArg(8, buffer_u_x_n1);
	cl_kernel_cal_dilatation.setArg(9, buffer_u_y_n1);
	cl_kernel_cal_dilatation.setArg(10, buffer_u_z_n1);
	cl_kernel_cal_dilatation.setArg(11, buffer_iter_neighbor_list);
	cl_kernel_cal_dilatation.setArg(12, buffer_node);
	cl_kernel_cal_dilatation.setArg(13, buffer_small_delta);
	
	cl::Kernel cl_kernel_cal_displacement(clProgram, "cal_displacement");
		
	cl_kernel_cal_displacement.setArg(0, buffer_f_x);
	cl_kernel_cal_displacement.setArg(1, buffer_f_y);
	cl_kernel_cal_displacement.setArg(2, buffer_f_z);
	cl_kernel_cal_displacement.setArg(3, buffer_b_x);
	cl_kernel_cal_displacement.setArg(4, buffer_b_y);
	cl_kernel_cal_displacement.setArg(5, buffer_b_z);
	cl_kernel_cal_displacement.setArg(6, buffer_u_dot_x_nhalf);
	cl_kernel_cal_displacement.setArg(7, buffer_u_dot_y_nhalf);
	cl_kernel_cal_displacement.setArg(8, buffer_u_dot_z_nhalf);
	cl_kernel_cal_displacement.setArg(9, buffer_u_x_n0);
	cl_kernel_cal_displacement.setArg(10, buffer_u_y_n0);
	cl_kernel_cal_displacement.setArg(11, buffer_u_z_n0);
	cl_kernel_cal_displacement.setArg(12, buffer_u_x_n1);
	cl_kernel_cal_displacement.setArg(13, buffer_u_y_n1);
	cl_kernel_cal_displacement.setArg(14, buffer_u_z_n1);
	cl_kernel_cal_displacement.setArg(15, buffer_u_dot_x_n0);
	cl_kernel_cal_displacement.setArg(16, buffer_u_dot_y_n0);
	cl_kernel_cal_displacement.setArg(17, buffer_u_dot_z_n0);
	cl_kernel_cal_displacement.setArg(18, buffer_u_doubledot_x_n0);
	cl_kernel_cal_displacement.setArg(19, buffer_u_doubledot_y_n0);
	cl_kernel_cal_displacement.setArg(20, buffer_u_doubledot_z_n0);
	cl_kernel_cal_displacement.setArg(21, buffer_ro);
	cl_kernel_cal_displacement.setArg(22, buffer_delta_t);
	
	//Main kernel
	ofstream file_17;
	file_17.open ("disp_cpp.txt");
	for (size_t t_step = 0; t_step < num_steps; ++t_step){
		if(t_step%50==0){
			cout<<"Time step t=" <<t_step<< endl;
		}
		//First partial velocity update & nodal displacement
		
		queue.enqueueWriteBuffer(buffer_u_dot_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_dot_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_dot_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_doubledot_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_doubledot_x_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_doubledot_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_doubledot_y_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_doubledot_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_doubledot_z_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_x_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_y_n0[0]);
		queue.enqueueWriteBuffer(buffer_u_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_z_n0[0]);
		
		cl::Event evt03;
		queue.enqueueNDRangeKernel(cl_kernel_u_reinitialization, cl::NullRange, globalRange, cl::NullRange, NULL, &evt03);
		
		queue.enqueueReadBuffer(buffer_u_dot_x_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_nhalf[0]);
		queue.enqueueReadBuffer(buffer_u_dot_y_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_nhalf[0]);
		queue.enqueueReadBuffer(buffer_u_dot_z_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_nhalf[0]);
		queue.enqueueReadBuffer(buffer_u_x_n1, CL_TRUE, 0, sizeof(float) * node, &u_x_n1[0]);
		queue.enqueueReadBuffer(buffer_u_y_n1, CL_TRUE, 0, sizeof(float) * node, &u_y_n1[0]);
		queue.enqueueReadBuffer(buffer_u_z_n1, CL_TRUE, 0, sizeof(float) * node, &u_z_n1[0]);
		
		queue.finish();
		
		//Apply BC
		for (size_t j = 0; j <4*(ndivy*ndivz); ++j) { // in the beginning of the block
			u_x_n0[j]=0.0; 	u_y_n0[j]=0.0;	u_z_n0[j]=0.0;
			u_dot_x_n0[j]=0.0; 	u_dot_y_n0[j]=0.0;	u_dot_z_n0[j]=0.0;
			u_doubledot_x_n0[j]=0.0; u_doubledot_y_n0[j]=0.0; u_doubledot_z_n0[j]=0.0;
			u_dot_x_nhalf[j]=0.0; 	u_dot_y_nhalf[j]=0.0;	u_dot_z_nhalf[j]=0.0;
			u_x_n1[j]=0.0;          u_y_n1[j]=0.0;          u_z_n1[j]=0.0;
		}
			
		//Compute the dilatation using u at (n+1)
		
		queue.enqueueWriteBuffer(buffer_u_x_n1, CL_TRUE, 0, sizeof(float) * node, &u_x_n1[0]);
		queue.enqueueWriteBuffer(buffer_u_y_n1, CL_TRUE, 0, sizeof(float) * node, &u_y_n1[0]);
		queue.enqueueWriteBuffer(buffer_u_z_n1, CL_TRUE, 0, sizeof(float) * node, &u_z_n1[0]);
		
		cl::Event evt_04;
		queue.enqueueNDRangeKernel(cl_kernel_cal_dilatation, cl::NullRange, globalRange, cl::NullRange, NULL, &evt_04);
		
		queue.enqueueReadBuffer(buffer_theta, CL_TRUE, 0, sizeof(float) * node, &theta[0]);
				
		queue.finish();
		
		//Re-initialization peridynamics force		
		cl::Event evt05;
		queue.enqueueNDRangeKernel(cl_kernel_pforce_reinitialization, cl::NullRange, globalRange, cl::NullRange, NULL, &evt05);
		
		queue.enqueueReadBuffer(buffer_f_x, CL_TRUE, 0, sizeof(float) * node, &f_x[0]);
		queue.enqueueReadBuffer(buffer_f_y, CL_TRUE, 0, sizeof(float) * node, &f_y[0]);
		queue.enqueueReadBuffer(buffer_f_z, CL_TRUE, 0, sizeof(float) * node, &f_z[0]);		
		
		queue.finish();
		
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
		
		//Calculate displacement
		queue.enqueueWriteBuffer(buffer_f_x, CL_TRUE, 0, sizeof(float) * node, &f_x[0]);
		queue.enqueueWriteBuffer(buffer_f_y, CL_TRUE, 0, sizeof(float) * node, &f_y[0]);
		queue.enqueueWriteBuffer(buffer_f_z, CL_TRUE, 0, sizeof(float) * node, &f_z[0]);
		queue.enqueueWriteBuffer(buffer_u_dot_x_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_nhalf[0]);
		queue.enqueueWriteBuffer(buffer_u_dot_y_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_nhalf[0]);
		queue.enqueueWriteBuffer(buffer_u_dot_z_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_nhalf[0]);
		//write buffer_u_x_n1 is already executed above
		
		
		cl::Event evt07;
		queue.enqueueNDRangeKernel(cl_kernel_cal_displacement, cl::NullRange, globalRange, cl::NullRange, NULL, &evt07);
		
		queue.enqueueReadBuffer(buffer_u_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_x_n0[0]);
		queue.enqueueReadBuffer(buffer_u_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_y_n0[0]);
		queue.enqueueReadBuffer(buffer_u_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_z_n0[0]);
		queue.enqueueReadBuffer(buffer_u_dot_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_n0[0]);
		queue.enqueueReadBuffer(buffer_u_dot_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_n0[0]);
		queue.enqueueReadBuffer(buffer_u_dot_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_n0[0]);
		queue.enqueueReadBuffer(buffer_u_doubledot_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_doubledot_x_n0[0]);
		queue.enqueueReadBuffer(buffer_u_doubledot_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_doubledot_y_n0[0]);
		queue.enqueueReadBuffer(buffer_u_doubledot_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_doubledot_z_n0[0]);
		
		queue.finish();
		
		file_17 <<t_step<<"   "<<u_x_n1[floor(node/2)]<<"   "<<u_x_n1[node-1]<<endl; //disp at end of rope
		
	} //end of time integration
	file_17.close();
	
	system_clock::time_point stop_parallel = system_clock::now();
	std::chrono::duration<float, std::milli> duration_parallel = stop_parallel - start_parallel;
	cout << "Parallel peridynamics = "<<duration_parallel.count()<<" millisecond"<<endl;
	
	
	
	for (size_t i = 0; i < node; ++i) {
		u_n1[i]=sqrt(pow(u_x_n1[i],2.0)+pow(u_y_n1[i],2.0)+pow(u_z_n1[i],2.0));
		x_plus_ux[i]=x[i]+u_x_n1[i];
		y_plus_uy[i]=y[i]+u_y_n1[i];
		z_plus_uz[i]=z[i]+u_z_n1[i];
	}
	
	
	ofstream file_18;
	file_18.open ("pos_vs_disp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_18 <<x[i]<<"   "<<u_x_n1[i]<<"   "<<u_y_n1[i]<<"   "<<u_z_n1[i]<<"   "<<u_n1[i];
		if(i < (x_plus_ux.size()-1)) {
			file_18 <<endl;
		}
		
	}
	file_18.close();

	printf("End of program!");
	
	
}