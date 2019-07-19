__kernel void algo1(const float small_delta, __global float *xi, \
					__global const float *delta_V, __global float *m,\
					__global const float *x)
{ 
    int i = get_global_id(0);
	
	m[i]=x[i]*4.0;
	xi[i]=x[i]*4.0;
	/*for (size_t k = 0; k < NEIGHBOR_LIST; ++k){     //NEIGHBOR_LIST=neighbor_list.size()
		size_t j=neighbor_list[k];
		std::vector<float> xi(node,0.0);
			for (size_t idx = 0; idx < xi.size(); ++idx){	//xi= vector
				xi[idx]=(x[j]-x[i]);
			}
				float omega=exp(-abs(dot_product(xi,xi))/(small_delta*small_delta));
				m[i]=m[i]+omega*(dot_product(xi,xi))*delta_V[j];
	}
	*/
}