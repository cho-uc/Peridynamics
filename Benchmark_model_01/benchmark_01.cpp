#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <fstream> // for writing to file
#include <math.h>       //exp, pi
#include "tensor_algebra.h"

using namespace std;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	const float E = 200.0e9; // Young's modulus
	const float nu=0.25; //Poisson's ratio
	const float mu=E/(2*(1+nu)); //shear modulus
	const float k=E/(3*(1-2*nu)); // bulk modulus
	const float ro=7850; // mass densiy
	const float load = 200.0; //Newton
	
	const float length = 1.0; //X
	const float width = 1.0e-3; //Y
	const float heigth = 1.0e-3; //Z
	
	//const size_t ndivx = 1000;
	const size_t ndivx = 100;
	const size_t ndivy = 1;
	const size_t ndivz = 1;
	const size_t node = ndivx*ndivy*ndivz;
	const float delta=length/ndivx;
	const float small_delta=3.015*delta; //horizon
	
	//const float delta_t_critical=(length/ndivx)/sqrt(k/ro);
	//const float delta_t=0.7*delta_t_critical;
	const float delta_t=1.94598e-7;
	//const float T=(26000.0*delta_t);
	const float T=(300.0*delta_t);
	const float num_steps= T/delta_t;
	
	
	
	vector<float> m(node,0.0); // weight
	vector<float> x(node,0.0);
	vector<float> y(node,0.0);
	vector<float> z(node,0.0);
	
	vector<float> f_x(node,0.0); //TODO : f_x should be separated between n & (n+1)?
	vector<float> f_y(node,0.0);
	vector<float> f_z(node,0.0);
	
	vector<float> b_x(node,0.0); //body force
	vector<float> b_y(node,0.0);
	vector<float> b_z(node,0.0);
	
	vector< vector<float>> d(node, vector<float>(node,0.0)); //damage variable
	vector< vector<size_t>> neighbor_list;
	vector<float> theta(node,0.0);	//dilation
	vector<float> delta_V(node,0.0);	//nodal Volume
	
	
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
	
	cout << "delta_t = "<<delta_t<<endl;
	
	for (size_t i = 0; i < ndivx; ++i) {
		for (size_t j = 0; j < ndivy; ++j) {
			for (size_t k = 0; k < ndivz; ++k) {
				x[i]=(length/ndivx)*(0.5+i);
				y[i+j]=0.0;
				z[i+j+k]=0.0;
				delta_V[i+j+k]=(length/ndivx)*(width/ndivy)*(heigth/ndivz);
			}
		}		
	}
	ofstream file_14;
	file_14.open ("coord_cpp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_14 <<i<< "   "<<x[i]<<"   "<<y[i]<<"   "<<z[i]<<"   "<<delta_V[i]<<endl;
	}
	file_14.close();
	
	for(size_t i = 0; i < node; ++i){
		neighbor_list.push_back(vector<size_t>());
		for(size_t j = 0; j < node; ++j){
			if (i!=j){
				float distance =sqrt(pow((x[i]-x[j]),2)+pow((y[i]-y[j]),2)+pow((z[i]-z[j]),2));
				if (distance<small_delta){
					neighbor_list[i].push_back(j);
				}
			}
		}
	}
	
	ofstream file_15;
	file_15.open ("neighbor_list_cpp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_15 <<i<< "-->";
		for (size_t j = 0; j < neighbor_list[i].size(); ++j) {
			file_15 <<"   "<<neighbor_list[i][j];
		}
		file_15 <<endl;
	}
	file_15.close();
	
	//Initial condition
	b_x[ndivx-1] = 0.0; //load at the end node
	for (size_t i = 0; i < node; ++i) {
		u_x_n0[i]=0.001*x[i];
	}
	
	ofstream file_18;
	file_18.open ("disp_init_cpp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_18 <<i<<"   "<<u_x_n0[i]<<endl;
	}
	file_18.close();
	
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
				m[i]+=(omega*xi_square*delta_V[j]);
			}
			
			
	}

	ofstream file_16;
	file_16.open ("weighted_vol_cpp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_16 <<i<<"   "<<m[i]<<endl;
	}
	file_16.close();
	
	//###########################################################################
	//Main kernel
	
	ofstream file_17;
	file_17.open ("disp_cpp.txt");
	for (size_t t_step = 0; t_step < num_steps; ++t_step){
		cout<<"==========================" << endl;
		cout<<"Time step t=" <<t_step<< endl;
		
		//Compute the dilatation
		for (size_t i = 0; i < node; ++i) {
			f_x[i]=0.0; //TODO : Re-initialization here ???
			f_y[i]=0.0;
			f_z[i]=0.0;
			
			theta[i]=0.0;
			for (size_t k = 0; k < neighbor_list[i].size(); ++k){
				size_t j=neighbor_list[i][k];
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				
				float eta_x=u_x_n1[j]-u_x_n1[i]; //TODO : u(n) or u(n+1)??
				float eta_y=u_y_n1[j]-u_y_n1[i];
				float eta_z=u_z_n1[j]-u_z_n1[i];
				float xi_square=pow(xi_x,2)+pow(xi_y,2)+pow(xi_z,2);
				float omega=exp(-xi_square/(small_delta*small_delta));
				float xi_plus_eta=sqrt(pow((xi_x+eta_x),2)+pow((xi_y+eta_y),2)+pow((xi_z+eta_z),2));
				float e=xi_plus_eta-sqrt(xi_square);	//extension state			
				theta[i]+=(3.0/m[i]*omega*(sqrt(xi_square))*e*delta_V[j]);
				
			}
		}
		//Compute the pairwise contributions to the global force density vector
		for (size_t i = 0; i < node; ++i) {
			for (size_t k = 0; k < neighbor_list[i].size(); ++k){
				size_t j=neighbor_list[i][k];
				
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
				float t=(3.0/m[i]*k*theta[i]*omega*sqrt(xi_square)) + (15.0*mu/m[i]*omega*e_d);
				float M_x=(xi_x+eta_x)/xi_plus_eta;
				float M_y=(xi_y+eta_y)/xi_plus_eta;
				float M_z=(xi_z+eta_z)/xi_plus_eta;
				
				f_x[i] += (t*M_x*delta_V[j]);
				f_y[i] += (t*M_y*delta_V[j]);
				f_z[i] += (t*M_z*delta_V[j]);
				
				f_x[j] -= (t*M_x*delta_V[i]);
				f_y[j] -= (t*M_y*delta_V[i]);
				f_z[j] -= (t*M_z*delta_V[i]);
			}
			
		}
		
		//Calculate displacement
		for (size_t i = 0; i < node; ++i) {
			
			u_dot_x_nhalf[i]=u_dot_x_n0[i]+(delta_t/2.0*u_doubledot_x_n0[i]);
			u_dot_y_nhalf[i]=u_dot_y_n0[i]+(delta_t/2.0*u_doubledot_y_n0[i]);
			u_dot_z_nhalf[i]=u_dot_z_n0[i]+(delta_t/2.0*u_doubledot_z_n0[i]);
			
			
			u_x_n1[i]=u_x_n0[i]+(delta_t*u_dot_x_nhalf[i]);
			u_y_n1[i]=u_y_n0[i]+(delta_t*u_dot_y_nhalf[i]);
			u_z_n1[i]=u_z_n0[i]+(delta_t*u_dot_z_nhalf[i]);
			
			//Apply BC : Not applicable
			
			
			u_doubledot_x_n1[i]=(f_x[i]+b_x[i])/ro;
			u_doubledot_y_n1[i]=(f_y[i]+b_y[i])/ro;
			u_doubledot_z_n1[i]=(f_z[i]+b_z[i])/ro;
			//cout<<"u_doubledot_x_n1 = "<<u_doubledot_x_n1 [i]<<", for i="<<i<<endl;
			
			u_dot_x_n1[i]=u_dot_x_nhalf[i]+(delta_t/2.0*u_doubledot_x_n1[i]);
			u_dot_y_n1[i]=u_dot_y_nhalf[i]+(delta_t/2.0*u_doubledot_y_n1[i]);
			u_dot_z_n1[i]=u_dot_z_nhalf[i]+(delta_t/2.0*u_doubledot_z_n1[i]);
			
			//Re-initialization for (n+1) time step
			u_x_n0[i]=u_x_n1[i];
			u_y_n0[i]=u_y_n1[i];
			u_z_n0[i]=u_z_n1[i];
			u_dot_x_n0[i]=u_dot_x_n1[i];
			u_dot_y_n0[i]=u_dot_y_n1[i];
			u_dot_z_n0[i]=u_dot_z_n1[i];
			u_doubledot_x_n0[i]=u_doubledot_x_n1[i];
			u_doubledot_y_n0[i]=u_doubledot_y_n1[i];
			u_doubledot_z_n0[i]=u_doubledot_z_n1[i];
			
			//cout<<"u_x_n1 = "<<u_x_n1 [i]<<", for i="<<i<<endl;
		}
		
		file_17 <<t_step<<"   "<<u_x_n1[node-1]<<endl; //disp at end of rope
		
	} //end of time integration
	file_17.close();
	
	printf("End of program!");
	
	
	
	
}