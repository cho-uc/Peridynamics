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
	
	float mu=1.2; //shear modulus
	float k=1.2; // bulk modulus
	
	float delta=0.2;
	float small_delta=3*delta; //horizon
	float G0=1.0;				// fracture energy per unit area
	float s0=sqrt(5*G0/(9*k*small_delta));
	size_t node = 100;
	
	vector<vector<int>>  neighbor_list(node, vector<int>(7,0.0)); //7 nodes
	vector<vector<int>>  traversal_list(node, vector<int>(7,0.0));
	vector<float> m(node,0.0); // weight
	vector<float> x(node,0.0);
	vector<float> y(node,0.0);
	vector<float> z(node,0.0);
	vector<float> f_x(node,0.0);
	vector<float> f_y(node,0.0);
	vector<float> f_z(node,0.0);
	
	vector<float> u_x_n0(node,0.0);	vector<float> u_x_n1(node,0.0);
	vector<float> u_y_n0(node,0.0);	vector<float> u_y_n1(node,0.0); 	
	vector<float> u_z_n0(node,0.0);	vector<float> u_z_n1(node,0.0); 
	
	vector< vector<double> > d(node, vector<double>(node,0.0)); //damage variable
	vector<float> theta(node,0.0);	//dilation
	vector<float> delta_V(node,0.0);	//nodal Volume
	
	
	vector<float> epsilon(node,0.0);	
	vector<float> M(node,0.0);	
	vector<float> u(node,0.0);
	for (size_t i = 0; i < 7; ++i) {
		//epsilon[i]=rand() % 7+ 1;
	}
	for (size_t i = 0; i < neighbor_list.size(); ++i) {
		for (size_t j = 0; j < neighbor_list[0].size(); ++j){
			//neighbor_list[i][j]=rand() % node +round(3*j+i/2);
			neighbor_list[i][j]=3*j+round(i/2);
		}
	}
	
	for (size_t i = 0; i < node; ++i) {
		x[i]=((float) rand()/(RAND_MAX));
		y[i]=((float) rand()/(RAND_MAX));
		z[i]=((float) rand()/(RAND_MAX));
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
	//printArray_f(m);
	
	//###########################################################################
	
	//Algo II (Linear Peridynamic Solid Internal Force)
	
	//Compute the dilatation
	for (size_t i = 0; i < node; ++i) {
		theta[i]=0.0;
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
			theta[i]=theta[i]+(3.0/m[i]*omega*(sqrt(xi_square))*e*delta_V[j]);
				
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
			float t=(3.0/m[i]*k*theta[i]*omega*sqrt(xi_square))+(15.0*mu/m[i]*omega*e_d);
			float M_x=(xi_x+eta_x)/xi_plus_eta;
			float M_y=(xi_y+eta_y)/xi_plus_eta;
			float M_z=(xi_z+eta_z)/xi_plus_eta;
			
			f_x[i]=f_x[i]+t*M_x*delta_V[j];
			f_y[i]=f_y[i]+t*M_y*delta_V[j];
			f_z[i]=f_z[i]+t*M_z*delta_V[j];
			f_x[j]=f_x[j]-t*M_x*delta_V[i];
			f_y[j]=f_y[j]-t*M_y*delta_V[i];
			f_z[j]=f_z[j]-t*M_z*delta_V[i];
			
		}
	}
	
	
	//###########################################################################
	
	//Algo III (Critical Stretch Bond Failure)
	
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
			float xi_plus_eta=sqrt(pow((xi_x+eta_x),2)+pow((xi_y+eta_y),2)+pow((xi_z+eta_z),2));
			float s=(xi_plus_eta-sqrt(xi_square))/sqrt(xi_square);
			if (s>=s0){
				d[i][j]=1.0;
			}
		
		}
	
	}
	
	//###########################################################################
	/*
	//Algo IV (Tangent Stiffness Matrix)
	
	vector< vector<double> > K(node, vector<double>(node,0.0));
	vector<float> T_positive_dens(node,0.0);	//force densities per unit vol
	vector<float> T_negative_dens(node,0.0);	//force densities per unit vol
	vector<float> f_difference(node,0.0);
	
	
	for (size_t i = 0; i < node; ++i) {
		for (size_t k = 0; k < traversal_list.size(); ++k){
			size_t j=traversal_list[k];
			
			vector<float> T_positive(node,0.0);	//force
			vector<float> T_negative(node,0.0);	//force
			for (size_t r = 0; r < 2; ++r){ //max size of r?
				T_positive=add(u,epsilon);
				T_negative=substract(u,epsilon);
				//....
			
				
				vector<float> f_positive(node,0.0);	//force
				vector<float> f_negative(node,0.0);	//force
				
				
				for (size_t idx = 0; idx < neighbor_list.size(); ++idx){
					size_t k=neighbor_list[i];
					f_positive[k]=T_positive_dens[k]//*delta_V[i]*delta_V[k];
					f_negative[k]=T_negative_dens[k]*delta_V[i]*delta_V[k];
					f_difference[k]=f_positive[k]-f_negative[k];
					
					
					for (size_t s = 0; s < 4; ++s){ //DOF?
						K[s][r]=K[s][r]+f_difference[s]/(2.0*epsilon[s]);
					}
			
				}
			}
		}
	}
	*/
	
	//###########################################################################
	
	//Algo V (Short-Range Force Contact Algorithm)
	
	vector<float> f_contact_x(node,0.0);
	vector<float> f_contact_y(node,0.0);
	vector<float> f_contact_z(node,0.0);
	vector<float> x_deformed(node,0.0);
	vector<float> y_deformed(node,0.0);
	vector<float> z_deformed(node,0.0);
	
	float l0=1.5;
	float C=1.5;
	
	for (size_t i = 0; i < node; ++i) {
		x_deformed[i]=x[i]+u_x_n1[i];
		y_deformed[i]=y[i]+u_y_n1[i];
		z_deformed[i]=z[i]+u_z_n1[i];
		for (size_t k = 0; k < neighbor_list[i].size(); ++k){ //proximity=neighbor_list ?
			size_t j=neighbor_list[i][k];
			x_deformed[j]=x[j]+u_x_n1[j];
			y_deformed[j]=y[j]+u_y_n1[j];
			z_deformed[j]=z[j]+u_z_n1[j];
			float l=sqrt(pow((x_deformed[j]-x_deformed[i]),2)+pow((y_deformed[j]-y_deformed[i]),2)
					+pow((z_deformed[j]-z_deformed[i]),2));
			if(l<l0){
				float fc=9.0*C/(M_PI*(pow(small_delta,4)))*(l0-l)/small_delta;
				
				float M_x=(x_deformed[j]-x_deformed[i])/l;
				float M_y=(y_deformed[j]-y_deformed[i])/l;
				float M_z=(z_deformed[j]-z_deformed[i])/l;
				
				f_contact_x[i]=f_contact_x[i]-fc*delta_V[j]*M_x;
				f_contact_y[i]=f_contact_y[i]-fc*delta_V[j]*M_y;
				f_contact_z[i]=f_contact_z[i]-fc*delta_V[j]*M_z;
				
				
				f_contact_x[j]=f_contact_x[j]+fc*delta_V[i]*M_x; 
				f_contact_y[j]=f_contact_y[j]+fc*delta_V[i]*M_y;
				f_contact_z[j]=f_contact_z[j]+fc*delta_V[i]*M_z;
			
			}	
			
		}
	}
	
	cout<<"End of program"<<endl;
	
	//TODO :how to save neighbor_list, traversal_list (2D)
	//Q : How about 2D&3D Discretization?
	
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