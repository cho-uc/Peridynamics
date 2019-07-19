#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl2.hpp>
#endif

#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <math.h>       //exp, pi
#include "tensor_algebra.h"


 
int main(void){
	//SERIAL VERSION
	{
		float delta=0.2;
		float small_delta=3*delta; //horizon
		
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
		cout<<"norm (m_serial) = "<<(norm(m))<<endl;
		
		
	}
	
	
	
	//########################################################################################
	//Check device
	{
	//get platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		std::cout << "*********** No platforms found! Aborting...!" << std::endl;
		return 1;
	}

	std::cout << "*********** Listing available platforms:" << std::endl;
	for (size_t i = 0; i < platforms.size(); ++i)
		std::cout << "platform[" << i << "]: " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

	cl::Platform selectedPlatform = platforms[1]; // choose Intel or NVidia
	std::cout << "*********** Using the following platform: " << selectedPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	//get devices
	std::vector<cl::Device> devices;
	selectedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.empty())
	{
		std::cout << "*********** No devices found on platform " << selectedPlatform.getInfo<CL_PLATFORM_NAME>()
			<<"! Aborting...!" << std::endl;
		return 1;
	}

	std::cout << "*********** Listing available devices:" << std::endl;
	for (size_t i = 0; i < devices.size(); ++i)
		std::cout << "device[" << i << "]: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;

	cl::Device selectedDevice = devices[0]; // choose Intel or NVidia
	std::cout << "*********** Using the following device: " << selectedDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
	}
	
	//########################################################################################
	//PARALLEL VERSION
    
	//Initialization
	
	float delta=0.2;
	float small_delta=3*delta; //horizon
	
	size_t node = 100;
	vector<int> neighbor_list (7, 0.0);
	vector<int> traversal_list (7, 0.0);
	vector<float> m(node,0.0); // weight
	vector<float> x(node,0.0);
	vector<float> f(node,0.0);
	vector<float> xi(node,0.0); // for kernel
	
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
		
	//_______________________________________
	
	cl_int error_ret;
    std::ifstream inStream("kernel_algo1_i.cl");
	//std::ifstream inStream("kernel_algo1_i_j.cl");
    if (inStream.fail())
    {
        std::cout << "Failed to load kernel. Aborting..." << std::endl;
        return 1;
    }
	
    std::string kernelStr;

    inStream.seekg(0, std::ios::end);
    kernelStr.reserve(inStream.tellg());
    inStream.seekg(0, std::ios::beg);
    //C++11
    kernelStr.assign(std::istreambuf_iterator<char>(inStream), {});
    inStream.close();

 
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        std::cout << "*********** No platforms found! Aborting...!" << std::endl;
        return 1;
    }

    cl::Platform selectedPlatform = platforms[1]; //choose Intel or NVidia
    std::cout << "*********** Using the following platform: " << selectedPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    //get devices
    std::vector<cl::Device> devices;
    selectedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty())
    {
        std::cout << "*********** No devices found on platform " << selectedPlatform.getInfo<CL_PLATFORM_NAME>()
            << "! Aborting...!" << std::endl;
        return 1;
    }

    cl::Device selectedDevice = devices[0];
    std::cout << "*********** Using the following device: " << selectedDevice.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context({selectedDevice});

    cl::Program::Sources sources;
    sources.push_back({kernelStr.c_str(), kernelStr.length()});

    cl::Program clProgram(context, sources);
    if (clProgram.build({selectedDevice}) != CL_SUCCESS) {
        std::cout<<"Building error: " << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selectedDevice) << std::endl;
        return 1;
    }
    
    // Create memory buffers on the device for each vector 
	
    cl::Buffer buffer_small_delta(context, CL_MEM_READ_ONLY, sizeof(float) * 1);
    cl::Buffer buffer_delta_V(context, CL_MEM_READ_ONLY, sizeof(float) * node);
    cl::Buffer buffer_m(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_xi(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(float) * node);
    cl::CommandQueue queue(context, selectedDevice);

    queue.enqueueWriteBuffer(buffer_small_delta, CL_TRUE, 0, (sizeof(float) *1), &small_delta);
	queue.enqueueWriteBuffer(buffer_xi, CL_TRUE, 0, sizeof(float) * node, &xi[0]);
    queue.enqueueWriteBuffer(buffer_delta_V, CL_TRUE, 0, sizeof(float) * node, &delta_V[0]);
	queue.enqueueWriteBuffer(buffer_m, CL_TRUE, 0, sizeof(float) * node, &m[0]);
	queue.enqueueWriteBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * node, &x[0]);
	
    cl::Kernel kernel_algo1(clProgram, "algo1");

    size_t globalRange = node;
    size_t localRange = 20;
    
	//const float small_delta, __global float *xi, __global const float *delta_V, __global float *m,__global const float *x)
    kernel_algo1.setArg(0, buffer_small_delta);
	kernel_algo1.setArg(1, buffer_xi);
	kernel_algo1.setArg(2, buffer_delta_V);
	kernel_algo1.setArg(3, buffer_m);
	kernel_algo1.setArg(4, buffer_x);
    queue.enqueueNDRangeKernel(kernel_algo1, cl::NullRange, globalRange, localRange);
    queue.enqueueReadBuffer(buffer_m, CL_TRUE, 0, sizeof(float) * node, &m[0]);
    queue.finish();

    
    std::cout << "norm (m_parallel)= " << norm(m) << std::endl;
 
    return 0;
}