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
	vector<int> neighbor_list (7, 0.0);
	vector<int> traversal_list (7, 0.0);
	vector<float> m(node,0.0); // weight
	vector<float> x(node,0.0);
	vector<float> f(node,0.0);
	
	vector< vector<double> > d(node, vector<double>(node,0.0)); //damage variable
	vector<float> theta(node,0.0);	//dilation
	vector<float> delta_V(node,0.0);	//nodal Volume
	
	
	vector<float> epsilon(node,0.0);	
	vector<float> M(node,0.0);	
	vector<float> u(node,0.0);
	for (size_t i = 0; i < 7; ++i) {
		neighbor_list[i]=rand() % 7+ 1;
		traversal_list[i]=rand() % 7+ 1;
		epsilon[i]=rand() % 7+ 1;
	}
	for (size_t i = 0; i < node; ++i) {
		x[i]=((float) rand()/(RAND_MAX));
		delta_V[i]=0.1;
	}

	//###########################################################################
	//Algo I (Linear Peridynamic Solid Initialization)

	for (size_t i = 0; i < node; ++i) {
			m[i]=0;
			for (size_t k = 0; k < neighbor_list.size(); ++k){
				size_t j=neighbor_list[k];
				vector<float> xi(node,0.0);
				for (size_t idx = 0; idx < xi.size(); ++idx){	//xi= vector
					xi[idx]=(x[j]-x[i]);
				}
				float omega=exp(-abs(dot_product(xi,xi))/(small_delta*small_delta));
				m[i]=m[i]+omega*(dot_product(xi,xi))*delta_V[j];
			}
			
	}
	//printArray_f(m);
	
	//###########################################################################
	
	//Algo II (Linear Peridynamic Solid Internal Force)
	
	
	for (size_t i = 0; i < node; ++i) { //already initialized above?
		f[i]=0.0;
	}
	//Compute the dilatation
	for (size_t i = 0; i < node; ++i) {
		theta[i]=0.0;
		for (size_t k = 0; k < neighbor_list.size(); ++k){
			size_t j=neighbor_list[k];
			vector<float> xi(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//xi= vector
				xi[idx]=(x[j]-x[i]);
			}
			vector<float> eta(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//xi= vector
				eta[idx]=(u[j]-u[i]);
			}
			float omega=exp(-abs(dot_product(xi,xi))/(delta*delta));
			//float e=(norm((vector<float>&)add(xi,eta)))-norm(eta);	//extension state
			vector<float> xi_plus_eta=add(xi,eta);
			float e=(norm(xi_plus_eta))-(norm(eta));	//extension state
			theta[i]=theta[i]+(3.0/m[i]*omega*((float)norm(eta))*e*delta_V[j]);
				
		}
		
		
	}
	//Compute the pairwise contributions to the global force density vector
	for (size_t i = 0; i < node; ++i) {
		for (size_t k = 0; k < neighbor_list.size(); ++k){
			size_t j=neighbor_list[k];
			vector<float> xi(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//xi= vector
				xi[idx]=(x[j]-x[i]);
			}
			vector<float> eta(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//eta= vector
				eta[idx]=(u[j]-u[i]);
			}
			float omega=exp(-abs(dot_product(xi,xi))/(delta*delta));			
			//float e=norm((&vector<float>) add(xi,eta))-norm(eta);	//extension state
			vector<float> xi_plus_eta=add(xi,eta);
			float e=(norm(xi_plus_eta))-(norm(eta));	//extension state
			float e_d=e-(theta[i]*norm(eta)/3.0);	//deviatoric extension state
			float t=(3.0/m[i]*k*theta[i]*norm(eta))+(15.0*mu/m[i]*omega*e_d);
			vector<float> M=add(xi,eta);
			for (size_t idx = 0; idx < xi.size(); ++idx){
				M[idx]=M[idx]*norm(xi_plus_eta);
				//f[i]=f[i]+t*M*delta_V[j];
				//f[j]=f[j]-t*M*_V[i];
			}
			
		}
	}
	
	
	//###########################################################################
	
	//Algo III (Critical Stretch Bond Failure)
	
	for (size_t i = 0; i < node; ++i) {
		for (size_t k = 0; k < neighbor_list.size(); ++k){
			size_t j=neighbor_list[k];
			vector<float> xi(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//xi= vector
				xi[idx]=(x[j]-x[i]);
			}
			vector<float> eta(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//eta= vector
				eta[idx]=(u[j]-u[i]);
			}
			vector<float> xi_plus_eta=add(xi,eta);
			float s=(norm(xi_plus_eta)-norm(eta))/norm(eta);
			if (s>=s0){
				d[i][j]=1.0;
			}
		
		}
	
	}
	
	//###########################################################################
	
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
					/*f_positive[k]=T_positive_dens[k]//*delta_V[i]*delta_V[k];
					f_negative[k]=T_negative_dens[k]*delta_V[i]*delta_V[k];
					f_difference[k]=f_positive[k]-f_negative[k];
					
					
					for (size_t s = 0; s < 4; ++s){ //DOF?
						K[s][r]=K[s][r]+f_difference[s]/(2.0*epsilon[s]);
					}*/
			
				}
			}
		}
	}
	
	
	//###########################################################################
	
	//Algo V (Short-Range Force Contact Algorithm)
	
	vector<float> f_contact(node,0.0);
	vector<float> y(node,0.0);
	float l0=1.5;
	float C=1.5;
	
	for (size_t i = 0; i < node; ++i) {
		y[i]=x[i]+u[i];
		for (size_t k = 0; k < neighbor_list.size(); ++k){ //proximity=neighbor_list ?
			size_t j=neighbor_list[k];
			y[j]=x[j]+u[j];
			float l=abs(y[j]-y[i]);
			if(l<l0){
				float fc=9.0*C/(M_PI*(pow(small_delta,4)))*(l0-l)/small_delta;
				vector<float> M(node,0.0);
				for (size_t idx = 0; idx < M.size(); ++idx){
					M[idx]=(y[j]-y[i])/l;
				}
				f_contact[i]=f_contact[i]-fc*delta_V[j]*M[i];
				f_contact[j]=f_contact[j]-fc*delta_V[i]*M[i];
			
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