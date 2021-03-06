#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <math.h>       //exp, pi
#include "tensor_algebra.h"

using namespace std;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	const float nu = 0.25;//Poisson's ratio
	const float k=1.2; // bulk modulus
	const float E= 2.0e11; //Young's modulus
	const float ro=7850.0; // mass densiy
	
	const float delta_t=1.94598e-7;
	const size_t time_step=26000;
	const size_t T=time_step*delta_t;
	
	const float length=1;
	const size_t ndivx = 1000;
	const float dx = length / ndivx;
	
	const float small_delta=3.015*dx; //horizon
	size_t node = 1000;
	vector<size_t> neighbor_list (8*node, 0.0); // 8 points/horizon
	vector<float> m(node,0.0); // weight
	vector<float> x(node,0.0);
	vector<float> y(node,0.0);
	vector<float> z(node,0.0);
	
	vector<float> f_x(node,0.0); //TODO : f_x should be separated between n & (n+1)?
	vector<float> f_y(node,0.0);
	vector<float> f_z(node,0.0);
	
	vector<float> b_x(node,0.0);
	vector<float> b_y(node,0.0);
	vector<float> b_z(node,0.0);
	
	vector< vector<float> > d(node, vector<float>(node,0.0)); //damage variable
	vector<float> theta(node,0.0);	//dilation
	vector<float> delta_V(node,1.0e-9);	//nodal Volume
	
	
	vector<float> epsilon(node,0.0);	
	vector<float> M(node,0.0);	
	vector<float> u_x_n0(node,0.0);	vector<float> u_x_n1(node,0.0);
	vector<float> u_y_n0(node,0.0);	vector<float> u_y_n1(node,0.0); 	
	vector<float> u_z_n0(node,0.0);	vector<float> u_z_n1(node,0.0); 
	vector<float> u_dot_x_n0(node,0.0); vector<float> u_dot_x_nhalf(node,0.0); vector<float> u_dot_x_n1(node,0.0); 	
	vector<float> u_dot_y_n0(node,0.0); vector<float> u_dot_y_nhalf(node,0.0);vector<float> u_dot_y_n1(node,0.0); 
	vector<float> u_dot_z_n0(node,0.0);vector<float> u_dot_z_nhalf(node,0.0);vector<float> u_dot_z_n1(node,0.0);
	vector<float> u_doubledot_x_n0(node,0.0); vector<float> u_doubledot_x_n1(node,0.0); 	
	vector<float> u_doubledot_y_n0(node,0.0); vector<float> u_doubledot_y_n1(node,0.0); 
	vector<float> u_doubledot_z_n0(node,0.0);vector<float> u_doubledot_z_n1(node,0.0);
	
	for (size_t i = 0; i < 7; ++i) {
		neighbor_list[i]=rand() % node+ 1;
		epsilon[i]=rand() % 7+ 1;
	}
	
	for (size_t i = 0; i < node; ++i) {
		x[i]=(0.5*dx) + i* dx;
		y[i]=0.5*1.0e-3; // const	
		z[i]=0.5*1.0e-3; //const
	}
	for (size_t i = 0; i < node; ++i) {
		f_x[i]=((float) rand()/(RAND_MAX));
		f_y[i]=((float) rand()/(RAND_MAX));
		f_z[i]=((float) rand()/(RAND_MAX));
		b_x[i]=((float) rand()/(RAND_MAX));
		b_y[i]=((float) rand()/(RAND_MAX));
		b_z[i]=((float) rand()/(RAND_MAX));
		
		delta_V[i]=0.1;
	}

	//###########################################################################
	//Algo I (Linear Peridynamic Solid Initialization)

	for (size_t i = 0; i < node; ++i) {
			m[i]=0;
			for (size_t k = 0; k < neighbor_list[i].size(); ++k){
				size_t j=neighbor_list[i][k];
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				float xi_square=pow(xi_x,2)+pow(xi_y,2)+pow(xi_z,2);
				float omega=exp(-xi_square/(small_delta*small_delta));
				m[i]=m[i]+omega*xi_square*delta_V[j];
			}
			
	}

	
	//###########################################################################
	//Main kernel
	
	for (size_t i = 0; i < node; ++i) {
		for (size_t t = 0; t < T; t+=delta_t){
			u_dot_x_nhalf[i]=u_dot_x_n0[i]+(delta_t/2.0*u_doubledot_x_n0[i]);
			u_dot_y_nhalf[i]=u_dot_y_n0[i]+(delta_t/2.0*u_doubledot_y_n0[i]);
			u_dot_z_nhalf[i]=u_dot_z_n0[i]+(delta_t/2.0*u_doubledot_z_n0[i]);
			
			u_x_n1[i]=u_x_n0[i]+(delta_t*u_dot_x_nhalf[i]);
			u_y_n1[i]=u_y_n0[i]+(delta_t*u_dot_y_nhalf[i]);
			u_z_n1[i]=u_z_n0[i]+(delta_t*u_dot_z_nhalf[i]);
			
			//TODO : Apply BC
			//TODO : Calculate f
			u_doubledot_x_n1[i]=(f_x[i]+b_x[i])/ro;
			u_doubledot_y_n1[i]=(f_y[i]+b_y[i])/ro;
			u_doubledot_z_n1[i]=(f_z[i]+b_z[i])/ro;
			
			u_dot_x_n1[i]=u_dot_x_nhalf[i]+(delta_t/2.0*u_doubledot_x_n1[i]);
			u_dot_y_n1[i]=u_dot_y_nhalf[i]+(delta_t/2.0*u_doubledot_y_n1[i]);
			u_dot_z_n1[i]=u_dot_z_nhalf[i]+(delta_t/2.0*u_doubledot_z_n1[i]);
			
			//Re-initialization
			u_x_n0[i]=u_x_n1[i];
			u_x_n0[i]=u_x_n1[i];
			u_x_n0[i]=u_x_n1[i];
			u_dot_x_n0[i]=u_dot_x_n1[i];
			u_dot_y_n0[i]=u_dot_y_n1[i];
			u_dot_z_n0[i]=u_dot_z_n1[i];
			u_doubledot_x_n0[i]=u_doubledot_x_n1[i];
			u_doubledot_y_n0[i]=u_doubledot_y_n1[i];
			u_doubledot_z_n0[i]=u_doubledot_z_n1[i];
			
		
		}
	}
	
	cout<<"Total norm = "<<norm(u_x_n1)+norm(u_y_n1)+norm(u_z_n1)+norm(u_dot_x_n1)+norm(u_dot_y_n1)+norm(u_dot_z_n1)\
			+norm(u_doubledot_x_n1)+norm(u_doubledot_y_n1)+norm(u_doubledot_z_n1)<<endl;
	printf("End of program!");
	
	//TODO : Meshing
	
	/*
	
	vector<float> x1(5,0.0);
	vector<int> x2(5,0.0);
	vector<float> x3(5,0.0);
	vector<int> x4(5,0.0);
	
	for (size_t i = 0; i < 5; ++i) {
		x1[i]=7.3+i;
		x2[i]=i*2.2;
		x3[i]=i*1.2;
		x4[i]=i-4;
	}
	cout<<"x1 (float)= "<<endl;
	printArray(x1);
	cout<<"x2 (int)= "<<endl;
	printArray(x2);
	cout<<"x3 (float)= "<<endl;
	printArray(x3);
	cout<<"x4 (int)= "<<endl;
	printArray(x4);
	
	auto a=norm(x2);
	cout<<"Norm x2 = "<<a<<endl;
	
	
	*/
}