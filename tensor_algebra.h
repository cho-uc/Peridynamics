#include <cmath>	//for calculating power & NaN
#include<iostream>
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <math.h>       //exp, pi

using namespace std;
template<typename T>
void printArray(vector<T> &myArray) {
	for(size_t x = 0;x < myArray.size(); ++x){
		cout<<"  "<<myArray[x];
    }
	cout<<endl;
}

template<typename T>
void printMatrix(vector<vector<T>> &myArray) {
	for(size_t x = 0;x < myArray.size(); ++x){
        for(size_t y = 0;y < myArray[x].size();++y){
            cout<<myArray[x][y];
			printf("  ");
        }
        cout << endl;
    }		
}

template<typename T>
float dot_product(vector<T> &a,vector<T> &b) {
	T result=0.0;
	for(size_t i = 0;i < a.size(); ++i){
		result+=a[i]*b[i];
    }
	return result;
}

template<typename T>
vector<T> substract(vector<T> &a,vector<T> &b) {
	vector<T> result(a.size(),0.0);
	for(size_t i = 0;i < a.size(); ++i){
		result[i]=a[i]-b[i];
    }
	return result;
}

template<typename T>
vector<T> add(vector<T> &a,vector<T> &b) {
	vector<T> result(a.size(),0.0);
	for(size_t i = 0;i < a.size(); ++i){
		result[i]=a[i]+b[i];
    }
	return result;
}

template<typename T>
float norm(vector<T> &a) {
	float result=0.0;
	for(size_t i = 0;i < a.size(); ++i){
		result+=a[i]*a[i];
    }
	return (sqrt(result));
}

void fill_vector(std::vector<float> &vec){
  for (size_t i = 0; i < vec.size(); ++i)
    vec[i] = (float)rand() / RAND_MAX;
}