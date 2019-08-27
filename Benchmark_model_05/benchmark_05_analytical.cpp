/* Analytical solution to 'Block of Material under tension'
*/
#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <fstream> // for writing to file
#include <math.h>       //exp, pi



using namespace std;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	const float E = 200.0e9; // Young's modulus
	const float nu=0.25; //Poisson's ratio
	const float mu=E/(2.0*(1.0+nu)); //shear modulus
	const float k_bulk_mod=E/(3.0*(1.0-2.0*nu)); // bulk modulus
	const float ro=7850.0; // mass densiy
	
	
	const float length = 1.0; //X
	const float width = 1.0e-1; //Y
	const float heigth = 1.0e-1; //Z
	const float load = 200.0e6; //Newton
	
	const size_t ndivx = 100;
	//const size_t ndivx = 20;
	const size_t ndivy = 10;
	//const size_t ndivy = 2;
	const size_t ndivz = 10;
	//const size_t ndivz = 2;
	const size_t node = ndivx*ndivy*ndivz;
	const float delta=length/ndivx;
	const float small_delta=3.015*delta; //horizon
	const float delta_x=length/ndivx;
	const float delta_y=width/ndivy;
	const float delta_z=heigth/ndivz;
	
	vector<float> m(node,0.0); // weight
	vector<float> x;
	vector<float> y;
	vector<float> z;
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
	vector< vector<size_t>> neighbor_list;
	vector<float> theta(node,0.0);	//dilation
	vector<float> delta_V;	//nodal Volume
	
	
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
				x.push_back(delta_x*(0.5+i));
				y.push_back(delta_y*(0.5+j));
				z.push_back(delta_z*(0.5+k));
				delta_V.push_back(delta_x*delta_y*delta_z);
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
	
	
	
	for (size_t i = (node-(ndivy*ndivz)); i < node; ++i) {
		b_x[i] = load/delta_x; //load to end points
	}
	
	ofstream file_25;
	file_25.open ("bforce_cpp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_25 <<"   "<<b_x[i]<<endl;
	}
	file_25.close();
	
	
	for (size_t i = 0; i < node; ++i) {
		u_x_n1[i]=load/E*x[i];
		u_y_n1[i]=-nu*load/E*y[i];
		u_z_n1[i]=-nu*load/E*z[i];;
		u_n1[i]=sqrt(pow(u_x_n1[i],2.0)+pow(u_y_n1[i],2.0)+pow(u_z_n1[i],2.0));
		x_plus_ux[i]=x[i]+u_x_n1[i];
		y_plus_uy[i]=y[i]+u_y_n1[i];
		z_plus_uz[i]=z[i]+u_z_n1[i];
	}
	
	//Create ouput for VTK
	ofstream file_18;
	file_18.open ("pos_vs_disp.txt");
	for (size_t i = 0; i < node; ++i) {
		file_18 <<x_plus_ux[i]<<"   "<<y_plus_uy[i]<<"   "<<z_plus_uz[i]<<"   "<<u_n1[i];
		if(i < (x_plus_ux.size()-1)) {
			file_18 <<endl;
		}
	}
	
	file_18.close();
	
	printf("End of program!");
	
	
	
	
}