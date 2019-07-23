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

using namespace std;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	float mu=1.2; //shear modulus
	float k=1.2; // bulk modulus
	float ro=1.2; // mass densiy
	
	float delta_t=0.1;
	float T=1.0;
	size_t time_step=T/delta_t;
		//TODO : time_step can only run very small in kernel
	float delta=0.2;
	float small_delta=3*delta; //horizon
	float G0=1.0;				// fracture energy per unit area
	float s0=sqrt(5.0*G0/(9.0*k*small_delta));
	size_t node = 100;
	vector<int> neighbor_list (8*node, 0); // 8 points/horizon
	vector<int> neighbor_list_pointer (node, 0);
	vector<int> traversal_list (7, 0.0);
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
	
	vector< vector<double> > d(node, vector<double>(node,0.0)); //damage variable
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
	
	for (size_t i = 0; i < 7; ++i) {
		neighbor_list[i]=rand() % node+ 1;
		traversal_list[i]=rand() % node+ 1;
		epsilon[i]=rand() % 7+ 1;
	}
	for (size_t i = 0; i < node; ++i) {
		x[i]=((float) rand()/(RAND_MAX));
		y[i]=((float) rand()/(RAND_MAX));	
		z[i]=((float) rand()/(RAND_MAX));
		f_x[i]=((float) rand()/(RAND_MAX));
		f_y[i]=((float) rand()/(RAND_MAX));
		f_z[i]=((float) rand()/(RAND_MAX));
		b_x[i]=((float) rand()/(RAND_MAX));
		b_y[i]=((float) rand()/(RAND_MAX));
		b_z[i]=((float) rand()/(RAND_MAX));
		
		delta_V[i]=0.1;
	}
	
	//Serial variables
	vector<float> u_x_n0_s=u_x_n0;	vector<float> u_x_n1_s=u_x_n1;
	vector<float> u_y_n0_s=u_y_n0;	vector<float> u_y_n1_s=u_y_n1; 	
	vector<float> u_z_n0_s=u_z_n0;	vector<float> u_z_n1_s=u_z_n1; 
	vector<float> u_dot_x_n0_s=u_dot_x_n0; vector<float> u_dot_x_nhalf_s=u_dot_x_nhalf; vector<float> u_dot_x_n1_s=u_dot_x_n1; 	
	vector<float> u_dot_y_n0_s=u_dot_y_n0; vector<float> u_dot_y_nhalf_s=u_dot_y_nhalf; vector<float> u_dot_y_n1_s=u_dot_y_n1; 
	vector<float> u_dot_z_n0_s=u_dot_z_n0; vector<float> u_dot_z_nhalf_s=u_dot_z_nhalf; vector<float> u_dot_z_n1_s=u_dot_z_n1;
	vector<float> u_doubledot_x_n0_s=u_doubledot_x_n0; vector<float> u_doubledot_x_n1_s=u_doubledot_x_n1; 	
	vector<float> u_doubledot_y_n0_s=u_doubledot_y_n0; vector<float> u_doubledot_y_n1_s=u_doubledot_y_n1; 
	vector<float> u_doubledot_z_n0_s=u_doubledot_z_n0; vector<float> u_doubledot_z_n1_s=u_doubledot_z_n1;
	
	//###########################################################################
	//Main kernel serial
	
	
	for (size_t i = 0; i < node; ++i) {
		for (size_t j = 0; j < time_step; ++j){
			
			u_dot_x_nhalf_s[i]=u_dot_x_n0_s[i]+4.0;
			u_dot_y_nhalf_s[i]=u_dot_y_n0_s[i]+4.0;
			u_dot_z_nhalf_s[i]=u_dot_z_n0_s[i]+4.0;
			/*
			u_dot_x_nhalf_s[i]=u_dot_x_n0_s[i]+(delta_t/2.0*u_doubledot_x_n0_s[i]);
			u_dot_y_nhalf_s[i]=u_dot_y_n0_s[i]+(delta_t/2.0*u_doubledot_y_n0_s[i]);
			u_dot_z_nhalf_s[i]=u_dot_z_n0_s[i]+(delta_t/2.0*u_doubledot_z_n0_s[i]);
			
			
			u_x_n1_s[i]=u_x_n0_s[i]+(delta_t*u_dot_x_nhalf_s[i]);
			u_y_n1_s[i]=u_y_n0_s[i]+(delta_t*u_dot_y_nhalf_s[i]);
			u_z_n1_s[i]=u_z_n0_s[i]+(delta_t*u_dot_z_nhalf_s[i]);
			
			//TODO : Apply BC
			//TODO : Calculate f
			u_doubledot_x_n1_s[i]=(f_x[i]+b_x[i])/ro;
			u_doubledot_y_n1_s[i]=(f_y[i]+b_y[i])/ro;
			u_doubledot_z_n1_s[i]=(f_z[i]+b_z[i])/ro;
			
			u_dot_x_n1_s[i]=u_dot_x_nhalf_s[i]+(delta_t/2.0*u_doubledot_x_n1_s[i]);
			u_dot_y_n1_s[i]=u_dot_y_nhalf_s[i]+(delta_t/2.0*u_doubledot_y_n1_s[i]);
			u_dot_z_n1_s[i]=u_dot_z_nhalf_s[i]+(delta_t/2.0*u_doubledot_z_n1_s[i]);
			*/
		
		}
	}
	/*
	cout<<"Total norm serial= "<<norm(u_x_n1_s)+norm(u_y_n1_s)+norm(u_z_n1_s)+norm(u_dot_x_n1_s)+norm(u_dot_y_n1_s)+norm(u_dot_z_n1_s)\
			+norm(u_doubledot_x_n1_s)+norm(u_doubledot_y_n1_s)+norm(u_doubledot_z_n1_s)<<endl;
	*/
	cout<<"Total norm serial= "<<norm(u_dot_x_nhalf_s)+norm(u_dot_y_nhalf_s)+norm(u_dot_z_nhalf_s)<<endl;
			
	//###########################################################################
	//PARALLEL VERSION

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
	
	cl_int error_ret;
    std::ifstream inStream("kernel_main.cl");
	
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
	
    cl::Buffer buffer_u_dot_x_nhalf(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_u_dot_y_nhalf(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_u_dot_z_nhalf(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_u_dot_x_n0(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_u_dot_y_n0(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_u_dot_z_n0(context, CL_MEM_READ_WRITE, sizeof(float) * node);
	cl::Buffer buffer_time_step(context, CL_MEM_READ_ONLY, sizeof(float) *1);
	
    cl::CommandQueue queue(context, selectedDevice);

    queue.enqueueWriteBuffer(buffer_u_dot_x_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_nhalf[0]);
	queue.enqueueWriteBuffer(buffer_u_dot_y_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_nhalf[0]);
    queue.enqueueWriteBuffer(buffer_u_dot_z_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_nhalf[0]);
	queue.enqueueWriteBuffer(buffer_u_dot_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_n0[0]);
	queue.enqueueWriteBuffer(buffer_u_dot_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_n0[0]);
	queue.enqueueWriteBuffer(buffer_u_dot_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_n0[0]);
	queue.enqueueWriteBuffer(buffer_time_step, CL_TRUE, 0, sizeof(float) * 1, &time_step);
	
    cl::Kernel main_kernel(clProgram, "main_method");

    size_t globalRange = node;
    size_t localRange = 20;
    
    main_kernel.setArg(0, buffer_u_dot_x_nhalf);
	main_kernel.setArg(1, buffer_u_dot_y_nhalf);
	main_kernel.setArg(2, buffer_u_dot_z_nhalf);
	main_kernel.setArg(3, buffer_u_dot_x_n0);
	main_kernel.setArg(4, buffer_u_dot_y_n0);
	main_kernel.setArg(5, buffer_u_dot_z_n0);
	main_kernel.setArg(6, buffer_time_step);

    queue.enqueueNDRangeKernel(main_kernel, cl::NullRange, globalRange, localRange);
    queue.enqueueReadBuffer(buffer_u_dot_x_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_nhalf[0]);
	queue.enqueueReadBuffer(buffer_u_dot_y_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_nhalf[0]);
	queue.enqueueReadBuffer(buffer_u_dot_z_nhalf, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_nhalf[0]);
	queue.enqueueReadBuffer(buffer_u_dot_x_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_x_n0[0]);
	queue.enqueueReadBuffer(buffer_u_dot_y_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_y_n0[0]);
	queue.enqueueReadBuffer(buffer_u_dot_z_n0, CL_TRUE, 0, sizeof(float) * node, &u_dot_z_n0[0]);
    queue.finish();
	
	
	cout<<"Total norm parallel= "<<norm(u_dot_x_nhalf)+norm(u_dot_y_nhalf)+norm(u_dot_z_nhalf)<<endl;
	
	
	printf("End of program!");
	
	
	
	//TODO : Meshing
}