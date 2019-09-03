#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <fstream> // for writing to file
#include <math.h>       //exp, pi
#include "tensor_algebra.h"
#include <iomanip> // cout precision

using namespace std;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	const float E = 200.0e9; // Young's modulus
	const float nu=0.25; //Poisson's ratio
	const float mu=E/(2.0*(1.0+nu)); //shear modulus
	const float k_bulk_mod=E/(3.0*(1.0-2.0*nu)); // bulk modulus
	const float ro=7850.0; // mass densiy
	const float load = 200.0; //Newton
	
	const float length = 1.0; //X
	const float width = 1.0e-3; //Y
	const float heigth = 1.0e-3; //Z
	
	const size_t ndivx = 1000;
	//const size_t ndivx = 200;
	const size_t ndivy = 1;
	const size_t ndivz = 1;
	const size_t node = ndivx*ndivy*ndivz;
	const float delta=length/ndivx;
	const float small_delta=3.015*delta; //horizon
	
	
	vector<float> m(node,0.0); // weight
	vector<float> x(node,0.0);
	vector<float> y(node,0.0);
	vector<float> z(node,0.0);
	vector<float> x_plus_ux(node,0.0);
	vector<float> y_plus_uy(node,0.0);
	vector<float> z_plus_uz(node,0.0);
	
	vector<float> f_x(node,0.0); //TODO : f_x should be separated between n & (n+1)?
	vector<float> f_y(node,0.0);
	vector<float> f_z(node,0.0);
	
	vector<float> b_x(node,0.0); //body force
	vector<float> b_y(node,0.0);
	vector<float> b_z(node,0.0);
	
	vector< vector<float>> d(node, vector<float>(node,0.0)); //damage variable
	vector<size_t> neighbor_list(node*node/2,0); //assume length node*node/2
	vector<size_t> neighbor_list_pointer(node,0);
	vector<float> theta(node,0.0);	//dilation
	vector<float> delta_V(node,0.0);	//nodal Volume
	
	
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
	
	size_t iter_neighbor_list_pointer=0; 
	size_t iter_neighbor_list=0; //length of neighbor_list
	
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
	
	if(iter_neighbor_list==node*node/2){
		cout<<"neighbor_list array length exceeds the buffer"<<endl;
	}
	ofstream file_15;
	file_15.open ("neighbor_list_cpp.txt");
	file_15 <<"node --> neighbor_list_pointer"<<endl;
	for (size_t i = 0; i < node; ++i) {
		file_15 <<i<< "-->"<<neighbor_list_pointer[i]<<endl;
	}
	file_15 <<"============================================="<<endl;
	file_15 <<"neighbor_list"<<endl;
	for (size_t i = 0; i < iter_neighbor_list; ++i) {
		file_15 <<"   "<<neighbor_list[i] <<endl;
	}
	file_15.close();
	
	
	//Critical time step
	float V_dot_C=0.0;
	float V_dot_C_temp=0.0;
	for (size_t i = 0; i < node; ++i) {	
			V_dot_C_temp=0.0; //Re-initialization
			for (size_t k = neighbor_list_pointer[i]; k < neighbor_list_pointer[i+1]; ++k){
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
	const float delta_t=delta_t_critical*0.1; // safety factor = 1.0
	cout << "delta_t = "<<delta_t<<endl;
	
	const float T=(1100.0*delta_t);
	//const float T=(500.0*delta_t);
	cout<<"Total T="<<(T*1000.0)<<" ms"<<endl;
	const float num_steps= T/delta_t;
	
	
	for (size_t i = (node-4); i < node; ++i) {
		b_x[i] = load/(node*delta_V[i]); //load to end points
	}
	
	//###########################################################################
	//Algo I (Linear Peridynamic Solid Initialization)

	for (size_t i = 0; i < node; ++i) {
			m[i]=0;
			for (size_t k = neighbor_list_pointer[i]; k < neighbor_list_pointer[i+1]; ++k){
				size_t j=neighbor_list[k];
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				float xi_square=pow(xi_x,2.0)+pow(xi_y,2.0)+pow(xi_z,2.0);
				float omega=exp(-xi_square/(small_delta*small_delta));
				m[i]+=omega*xi_square*delta_V[j];
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
		
		if(t_step%50==0){
			cout<<"Time step t=" <<t_step<< endl;
		}
		
		//First partial velocity update & nodal displacement
		for (size_t i = 0; i < node; ++i) {
			u_dot_x_nhalf[i]=u_dot_x_n0[i]+(delta_t/2.0*u_doubledot_x_n0[i]);
			u_dot_y_nhalf[i]=u_dot_y_n0[i]+(delta_t/2.0*u_doubledot_y_n0[i]);
			u_dot_z_nhalf[i]=u_dot_z_n0[i]+(delta_t/2.0*u_doubledot_z_n0[i]);
			
			u_x_n1[i]=u_x_n0[i]+(delta_t*u_dot_x_nhalf[i]);
			u_y_n1[i]=u_y_n0[i]+(delta_t*u_dot_y_nhalf[i]);
			u_z_n1[i]=u_z_n0[i]+(delta_t*u_dot_z_nhalf[i]);
		}
		
		//Apply BC
		for (size_t j = 0; j <4; ++j) { // in the beginning of the string
			u_x_n0[j]=0.0; 	u_y_n0[j]=0.0;	u_z_n0[j]=0.0;
			u_dot_x_n0[j]=0.0; 	u_dot_y_n0[j]=0.0;	u_dot_z_n0[j]=0.0;
			u_doubledot_x_n0[j]=0.0; u_doubledot_y_n0[j]=0.0; u_doubledot_z_n0[j]=0.0;
			u_dot_x_nhalf[j]=0.0; 	u_dot_y_nhalf[j]=0.0;	u_dot_z_nhalf[j]=0.0;
			u_x_n1[j]=0.0;          u_y_n1[j]=0.0;          u_z_n1[j]=0.0;
		}
		
		//Compute the dilatation
		for (size_t i = 0; i < node; ++i) {
			theta[i]=0.0;
			for (size_t k = neighbor_list_pointer[i]; k < neighbor_list_pointer[i+1]; ++k){
				size_t j=neighbor_list[k];
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				
				float eta_x=u_x_n1[j]-u_x_n1[i];
				float eta_y=u_y_n1[j]-u_y_n1[i];
				float eta_z=u_z_n1[j]-u_z_n1[i];
				float xi_square=pow(xi_x,2.0)+pow(xi_y,2.0)+pow(xi_z,2.0); 
				float omega=exp(-xi_square/(small_delta*small_delta));
				float xi_plus_eta=sqrt(pow((xi_x+eta_x),2.0)+pow((xi_y+eta_y),2.0)+pow((xi_z+eta_z),2.0));
				float e=xi_plus_eta-sqrt(xi_square);	//extension state			
				theta[i]=theta[i]+(3.0/m[i]*omega*(sqrt(xi_square))*e*delta_V[j]);
				
			}
		}
		//Re-initialization peridynamics force
		for (size_t i = 0; i < node; ++i) {
			f_x[i]=0.0; 	f_y[i]=0.0; 	f_z[i]=0.0;
		}
		
		//Compute the pairwise contributions to the global force density vector
		for (size_t i = 0; i < node; ++i) {			
			for (size_t k = neighbor_list_pointer[i]; k < neighbor_list_pointer[i+1]; ++k){
				size_t j=neighbor_list[k];
				
				float xi_x=x[j]-x[i];
				float xi_y=y[j]-y[i];
				float xi_z=z[j]-z[i];
				
				float eta_x=u_x_n1[j]-u_x_n1[i];
				float eta_y=u_y_n1[j]-u_y_n1[i];
				float eta_z=u_z_n1[j]-u_z_n1[i];
				
				float xi_square=pow(xi_x,2.0)+pow(xi_y,2.0)+pow(xi_z,2.0);
				float omega=exp(-xi_square/(small_delta*small_delta)); // 0.6-0.8
				float xi_plus_eta=sqrt(pow((xi_x+eta_x),2.0)+pow((xi_y+eta_y),2.0)+pow((xi_z+eta_z),2.0));
				float e=xi_plus_eta-sqrt(xi_square);	//extension state			
				
				float e_d=e-(theta[i]*sqrt(xi_square)/3.0);	//deviatoric extension state
				float t=(3.0/m[i]*k_bulk_mod*theta[i]*omega*sqrt(xi_square))+(15.0*mu/m[i]*omega*e_d); //scalar force state
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
			
			u_doubledot_x_n1[i]=(f_x[i]+b_x[i])/ro;
			u_doubledot_y_n1[i]=(f_y[i]+b_y[i])/ro;
			u_doubledot_z_n1[i]=(f_z[i]+b_z[i])/ro;
			
			u_dot_x_n1[i]=u_dot_x_nhalf[i]+(delta_t/2.0*u_doubledot_x_n1[i]);
			u_dot_y_n1[i]=u_dot_y_nhalf[i]+(delta_t/2.0*u_doubledot_y_n1[i]);
			u_dot_z_n1[i]=u_dot_z_nhalf[i]+(delta_t/2.0*u_doubledot_z_n1[i]);
			
			//Re-initialization
			
			u_x_n0[i]=u_x_n1[i];
			u_y_n0[i]=u_y_n1[i];
			u_z_n0[i]=u_z_n1[i];
			u_dot_x_n0[i]=u_dot_x_n1[i];
			u_dot_y_n0[i]=u_dot_y_n1[i];
			u_dot_z_n0[i]=u_dot_z_n1[i];
			u_doubledot_x_n0[i]=u_doubledot_x_n1[i];
			u_doubledot_y_n0[i]=u_doubledot_y_n1[i];
			u_doubledot_z_n0[i]=u_doubledot_z_n1[i];
			
		}
		
		file_17 <<t_step<<"   "<<u_x_n1[floor(node/2)]<<"   "<<u_x_n1[node-4]<<endl; //disp at end of rope
		
	} //end of time integration
	file_17.close();
	
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
	
	//Create ouput for VTK
	ofstream file_19;
	file_19.open ("VTK_input.txt");
	for (size_t i = 0; i < node; ++i) {
		file_19 <<x_plus_ux[i]<<"   "<<y_plus_uy[i]<<"   "<<z_plus_uz[i]<<"   "<<u_n1[i];
		if(i < (x_plus_ux.size()-1)) {
			file_19 <<endl;
		}
	}
	file_19.close();
	
	printf("End of program!");
	
	
	
	
}