#include <cmath>	//for calculating power & NaN
#include <fstream>	//reading inputs
#include <sstream>	//reading inputs
#include <string>	//reading inputs
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <math.h>       //exp, pi
#include "tensor_algebra.h"

using namespace std;

int main(int argc, char **argv){
	vector<double> mesh (8*3,0.0);
	mesh[0]  = 0.0 ; mesh[1]  = 0.0 ; mesh[2]  = 0.0;
	mesh[3]  = 1.0 ; mesh[4]  = 0.0 ; mesh[5]  = 0.0;
	mesh[6]  = 0.0 ; mesh[7]  = 1.0 ; mesh[8]  = 0.0;
	mesh[9]  = 1.0 ; mesh[10] = 1.0 ; mesh[11] = 0.0;
	mesh[12] = 0.0 ; mesh[13] = 0.0 ; mesh[14] = 1.0;
	mesh[15] = 1.0 ; mesh[16] = 0.0 ; mesh[17] = 1.0;
	mesh[18] = 0.0 ; mesh[19] = 1.0 ; mesh[20] = 1.0;
	mesh[21] = 1.0 ; mesh[22] = 1.0 ; mesh[23] = 1.0;
	
//########################################################################################
	
	vector<float> x (70500,0.0);
	vector<float> y (70500,0.0);
	vector<float> z (70500,0.0);
	vector<float> delta_V (70500,0.0);
	
    std::fstream in("fragmenting_cylinder_cut.txt");
    std::string line;
    std::vector<std::vector<float>> imported_data;
    int i = 0;

    while (std::getline(in, line))
    {
        float value;
        std::stringstream ss(line);

        imported_data.push_back(std::vector<float>());

        while (ss >> value)
        {
            imported_data[i].push_back(value);
        }
        ++i;
    }
	
	for (size_t i = 0; i < imported_data.size(); ++i){
		for (size_t j = 0; j < imported_data[0].size(); ++j){
			x[i]=imported_data[i][0];
			y[i]=imported_data[i][1];
			z[i]=imported_data[i][2];
			delta_V[i]=imported_data[i][4];
		}
	}
	
   //Checking
   
   cout<<"Size of imported_data, i ="<<imported_data.size()<<" and j (0)= "<<imported_data[0].size()<<endl;
   cout<<"Size of imported_data, i ="<<imported_data.size()<<" and j (1)= "<<imported_data[1].size()<<endl;
   
   cout<<"imported_data= "<<endl;
	for (size_t i = 0; i < 5; ++i){
		for (size_t j = 0; j < 5; ++j){
		cout<<imported_data[i][j]<<"  ";
		}
		cout<<endl;
	}
	
	
	float x_total=0.0;
	float y_total=0.0;
	float z_total=0.0;
	float delta_V_total=0.0;
	cout<<"x= "<<endl;
	for (size_t i = 0; i < 5; ++i){
		cout<<x[i]<<"  ";
		x_total+=x[i];
	}
	cout<<endl;
	cout<<"x_total= "<<x_total<<endl;
	cout<<"y= "<<endl;
	for (size_t i = 0; i < 5; ++i){
		cout<<y[i]<<"  ";
		y_total+=y[i];
	}
	cout<<endl;
	cout<<"y_total= "<<y_total<<endl;
	cout<<"z= "<<endl;
	for (size_t i = 0; i < 5; ++i){
		cout<<z[i]<<"  ";
		z_total+=z[i];
	}
	cout<<endl;
	cout<<"z_total= "<<z_total<<endl;
	cout<<"delta_V= "<<endl;
	for (size_t i = 0; i < 5; ++i){
		cout<<delta_V[i]<<"  ";
		delta_V_total+=delta_V[i];
	}
	cout<<endl;
	cout<<"delta_V_total= "<<delta_V_total<<endl;
	
}