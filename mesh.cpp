#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <math.h>       //exp, pi
#include "tensor_algebra.h"

using namespace std;

int main(int argc, char **argv){
	mesh = vector<double>(8*3);
	mesh[0]  = 0.0 ; mesh[1]  = 0.0 ; mesh[2]  = 0.0;
	mesh[3]  = 1.0 ; mesh[4]  = 0.0 ; mesh[5]  = 0.0;
	mesh[6]  = 0.0 ; mesh[7]  = 1.0 ; mesh[8]  = 0.0;
	mesh[9]  = 1.0 ; mesh[10] = 1.0 ; mesh[11] = 0.0;
	mesh[12] = 0.0 ; mesh[13] = 0.0 ; mesh[14] = 1.0;
	mesh[15] = 1.0 ; mesh[16] = 0.0 ; mesh[17] = 1.0;
	mesh[18] = 0.0 ; mesh[19] = 1.0 ; mesh[20] = 1.0;
	mesh[21] = 1.0 ; mesh[22] = 1.0 ; mesh[23] = 1.0;
	
}