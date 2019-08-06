#include <cmath>	//for calculating power & NaN
#include<iostream>
#include <fstream> // for writing to file
#include<cstdio>
#include <vector>
#include <cstdlib>
#include <math.h>       //exp, pi


using namespace std;

int main(int argc, char **argv){
	cout<<"Start of program"<<endl;
	
	//const size_t ndivx = 100; //ndivx: Number of divisions in x direction - except boundary region	
	const size_t ndivx = 10; 
	const size_t ndivy = 10; //ndivy: Number of divisions in y direction - except boundary region
	const size_t ndivz = 10; //ndivz: Number of divisions in z direction - except boundary region
	const size_t nbnd = 3; //nbnd: Number of divisions in the boundary region
	const size_t totnode = (ndivx + nbnd)*ndivy*ndivz; //totnode: Total number of material points
	//const size_t nt = 400; //nt: Total number of time step
	const size_t nt = 40;
	const size_t maxfam = 100; //maxfam: Maximum number of material points inside a horizon of a material point
	size_t nodefam_length=10000000; //TODO : reduce size
	
	vector< vector<float> > coord(totnode, vector<float>(3,0.0));
	vector< vector<float> > pforce(totnode, vector<float>(3,0.0));
	vector< vector<float> > pforceold (totnode, vector<float>(3,0.0));
	vector< vector<float> > bforce(totnode, vector<float>(3,0.0));
	vector< vector<float> > stendens(totnode, vector<float>(3,0.0));
	vector< vector<float> > fncst(totnode, vector<float>(3,1.0)); //initialize to 1.0
	vector< vector<float> > fncstold(totnode, vector<float>(3,1.0)); //initialize to 1.0
	vector< vector<float> > disp(totnode, vector<float>(3,0.0));
	vector< vector<float> > vel(totnode, vector<float>(3,0.0));
	vector< vector<float> > velhalfold(totnode, vector<float>(3,0.0));
	vector< vector<float> > velhalf(totnode, vector<float>(3,0.0));
	vector< vector<float> > andisp(totnode, vector<float>(3,0.0));
	vector< vector<float> > acc(totnode, vector<float>(3,0.0));
	vector< vector<float> > massvec(totnode, vector<float>(3,0.0));
	
	vector<size_t> numfam(totnode,0);	//total count of family nodes
	vector<size_t> pointfam(totnode,0); //pointer to the 1st cell
	vector<size_t> nodefam(nodefam_length,0);	//node id of family nodes, 
	vector<size_t> alflag(totnode,0);

	float length = 1.0; //m
	float width = 0.1; //m
	float thick = 0.1; //m
	float dx = length / ndivx;
	float delta = 3.015 * dx; //delta: Horizon

	float tmpdx, tmpvol, tmpcx, tmpux, tmpcy, tmpuy, tmpcz, tmpuz, dmgpar1, dmgpar2, theta, phi ;
	float scr, scx, scy, scz, cn, cn1, cn2;
	size_t totint;
	
	
	float dens = 7850.0; //Density (kg/m3)
	float emod = 200.0e9; //Elastic modulus (N/m2)
	float pratio = 1.0/4.0; //pratio: Poisson's ratio
	float alpha = 23.0e-6; //alpha: Coefficient of thermal expansion
	float dtemp = 0.0; //dtemp: Temperature change
	float area = dx * dx;
	float vol = area * dx; //vol: Volume of a material point

	float bc = 12.0 * emod / (M_PI*pow(delta,4)); //bc: Bond constant 
	
	float sedload1 = 0.6 * emod * 1.0e-6 ; //sedload1: Strain energy density of a material point for the first loading condition
	float sedload2 = 0.6 * emod * 1.0e-6; //sedload2: Strain energy density of a material point for the second loading condition
	float sedload3 = 0.6 * emod * 1.0e-6; //sedload3: Strain energy density of a material point for the third loading condition

	float dt = 1.0;
	float totime = nt * dt; 
	float ctime = 0.0; //ctime: Current time
	float idist = 0.0; //idist: Initial distance
	float fac = 0.0; //fac: Volume correction factor
	float radij = dx / 2.0; //radij: Material point radius
	size_t nnum = 0; //nnum: Material point number
	size_t cnode = 0; //cnode: Current material point
	float nlength  = 0.0; //Length of deformed bond
	float dforce1 = 0.0; //dforce1: x component of the PD force between two material points
	float dforce2 = 0.0; //dforce2: y component of the PD force between two material points
	float dforce3 = 0.0; //dforce3: z component of the PD force between two material points
	float appres = 200.0e6; // Applied pressure


// Discretization of the blocks

	for (size_t i = 0; i < ndivz; ++i) {
		for (size_t j = 0; j < ndivy; ++j) {
			for (size_t k = 0; k < ndivx; ++k) {
				coord[nnum][0] = (dx /2.0) + k* dx;
				coord[nnum][1] = -1.0 /2.0 * width + (dx / 2.0) + j * dx;
				coord[nnum][2] = -1.0 /2.0 * thick + (dx / 2.0) + i * dx;
				if ((coord[nnum][0])>(length-dx)) {
					alflag[nnum] = 1; //TODO : ???
				}
				nnum = nnum + 1;
			}
		}
	
	}
	
	
	totint = nnum;
	
	// Discretization of the boundary region - left

	for (size_t i = 0; i < ndivz; ++i) {
		for (size_t j = 0; j < ndivy; ++j) {
			for (size_t k = 0; k < nbnd; ++k) {
				
				coord[nnum][0] = -(dx /2.0) - k* dx;
				coord[nnum][1] = -1.0 /2.0 * width + (dx / 2.0) + j * dx;
				coord[nnum][2] = -1.0 /2.0 * thick + (dx / 2.0) + i * dx;
				nnum = nnum + 1;
			}
		}
	
	}
	ofstream file_14;
	file_14.open ("coord_cpp.txt");
	for (size_t i = 0; i < totnode; ++i) {
		file_14 <<coord[i][0]<<"   "<<coord[i][1]<<"   "<<coord[i][2]<<endl;
	}
	file_14.close();
	
	//Neighbor list determination
	for (size_t i = 0; i < totnode; ++i) {
		if (i==0){
			pointfam[i]= 0;
		}
		else{
			pointfam[i] = pointfam[i-1] + numfam[i-1];
		}
		for (size_t j = 0; j < totnode; ++j) {
			if (i!=j) {
				float idist = sqrt(pow((coord[j][0]-coord[i][0]),2) + pow((coord[j][1]-coord[i][1]),2)+pow((coord[j][2]-coord[i][2]),2));
				if(idist<=delta){
					numfam[i] = numfam[i] + 1;
					nodefam[pointfam[i]+numfam[i]-1] = j;
				}
			}
		}
	}
	
	ofstream file_15;
	file_15.open ("pointfam_numfam_nodefam_cpp.txt");
	for (size_t i = 0; i < totnode+1000; ++i) {
		file_15 <<pointfam[i]<<"   "<<numfam[i]<<"   "<<nodefam[i]<<endl;
	} 
	file_15.close();
	
	//Surface correction factors
	//Loading 1 (x-direction)
	for (size_t i = 0; i < totnode; ++i) {
		disp[i][0] = 0.001*coord[i][0];
		disp[i][1] =0.0;
		disp[i][2] =0.0;
	}
	for (size_t i = 0; i < totnode; ++i) {
		stendens[i][0] = 0.0;
		for (size_t j = 0; j < numfam[i]; ++j) {
			cnode = nodefam[pointfam[i]+j];
			idist = sqrt(pow((coord[cnode][0]-coord[i][0]),2) + pow((coord[cnode][1]-coord[i][1]),2) + \
				pow((coord[cnode][2] - coord[i][2]),2));
			nlength = sqrt(pow((coord[cnode][0] + disp[cnode][0]-coord[i][0] - disp[i][0]),2) + \
				pow((coord[cnode][1] + disp[cnode][1] - coord[i][1] - disp[i][1]),2) + \
				pow((coord[cnode][2] + disp[cnode][2] - coord[i][2] - disp[i][2]),2));
			if (idist<(delta-radij)) {
				fac = 1.0;
			}
			else if(idist < (delta+radij)) {
				fac = (delta+radij-idist)/(2.0*radij);
			}
			else{
				fac = 0.0;
			}
				
			stendens[i][0] += 0.5*0.5*bc *(pow(((nlength-idist)/idist),2))*idist*vol*fac;
		}
    //Calculation of surface correction factor in x direction 
    //by finding the ratio of the analytical strain energy density value
    //to the strain energy density value obtained from PD Theory
		fncst[i][0] = sedload1 / stendens[i][0];
	}
	
	//Loading 2 (y-direction)
	for (size_t i = 0; i < totnode; ++i) {
		disp[i][0] =0.0;
		disp[i][1] = 0.001*coord[i][1];
		disp[i][2] =0.0;
	}
	for (size_t i = 0; i < totnode; ++i) {
		stendens[i][1] = 0.0;
		for (size_t j = 0; j < numfam[i]; ++j) {
			cnode = nodefam[pointfam[i]+j];
			idist = sqrt(pow((coord[cnode][0]-coord[i][0]),2) + pow((coord[cnode][1]-coord[i][1]),2) + \
				pow((coord[cnode][2] - coord[i][2]),2));
			nlength = sqrt(pow((coord[cnode][0] + disp[cnode][0]-coord[i][0] - disp[i][0]),2) + \
				pow((coord[cnode][1] + disp[cnode][1] - coord[i][1] - disp[i][1]),2) + \
				pow((coord[cnode][2] + disp[cnode][2] - coord[i][2] - disp[i][2]),2));
			if (idist<(delta-radij)) {
				fac = 1.0;
			}
			else if(idist < (delta+radij)) {
				fac = (delta+radij-idist)/(2.0*radij);
			}
			else{
				fac = 0.0;
			}
				
			stendens[i][1] += 0.5*0.5*bc *(pow(((nlength-idist)/idist),2))*idist*vol*fac;
		}
    //Calculation of surface correction factor in x direction 
    //by finding the ratio of the analytical strain energy density value
    //to the strain energy density value obtained from PD Theory
		fncst[i][1] = sedload1 / stendens[i][1];
	}
	
	//Loading 3 (z-direction)
	for (size_t i = 0; i < totnode; ++i) {
		disp[i][0] =0.0;
		disp[i][1] =0.0;
		disp[i][2] = 0.001*coord[i][2];
	}
	for (size_t i = 0; i < totnode; ++i) {
		stendens[i][2] = 0.0;
		for (size_t j = 0; j < numfam[i]; ++j) {
			cnode = nodefam[pointfam[i]+j];
			idist = sqrt(pow((coord[cnode][0]-coord[i][0]),2) + pow((coord[cnode][1]-coord[i][1]),2) + \
				pow((coord[cnode][2] - coord[i][2]),2));
			nlength = sqrt(pow((coord[cnode][0] + disp[cnode][0]-coord[i][0] - disp[i][0]),2) + \
				pow((coord[cnode][1] + disp[cnode][1] - coord[i][1] - disp[i][1]),2) + \
				pow((coord[cnode][2] + disp[cnode][2] - coord[i][2] - disp[i][2]),2));
			if (idist<(delta-radij)) {
				fac = 1.0;
			}
			else if(idist < (delta+radij)) {
				fac = (delta+radij-idist)/(2.0*radij);
			}
			else{
				fac = 0.0;
			}
				
			stendens[i][2] += 0.5*0.5*bc *(pow(((nlength-idist)/idist),2))*idist*vol*fac;
		}
    //Calculation of surface correction factor in x direction 
    //by finding the ratio of the analytical strain energy density value
    //to the strain energy density value obtained from PD Theory
		fncst[i][2] = sedload1 / stendens[i][2];
	}
	ofstream file_16;
	file_16.open ("fncst_cpp.txt");
	for (size_t i = 0; i < totnode; ++i) {
		file_16 <<fncst[i][0]<<"   "<<fncst[i][1]<<"   "<<fncst[i][2]<<endl;
	} 
	file_16.close();
	
	//Stable mass vector computation
	for (size_t i = 0; i < totnode; ++i) {
	//5 is a safety factor
		massvec[i][0] = 0.25* dt * dt * (4.0/3.0)*M_PI*(pow(delta,3)) * bc / dx; //* 5.0
		massvec[i][1] = 0.25* dt * dt * (4.0/3.0)*M_PI*(pow(delta,3)) * bc / dx ;//* 5.0
		massvec[i][2]= 0.25* dt * dt * (4.0/3.0)*M_PI*(pow(delta,3)) * bc / dx; //* 5.0
	
	}
	ofstream file_17;
	file_17.open ("massvec_cpp.txt");
	for (size_t i = 0; i < totnode; ++i) {
		file_17 <<massvec[i][0]<<"   "<<massvec[i][1]<<"   "<<massvec[i][2]<<endl;
	} 
	file_17.close();
	
	//Applied loading - Right
	for (size_t i = 0; i < totint; ++i) { // loop until totint
		if (alflag[i]==1) {
			bforce[i][0] = appres /dx;
		}
	}
	
	ofstream file_18;
	file_18.open ("bforce_cpp.txt");
	for (size_t i = 0; i < totnode; ++i) {
		file_18 <<bforce[i][0]<<"   "<<bforce[i][1]<<"   "<<bforce[i][2]<<endl;
	} 
	file_18.close();
	
	cout<<"Got here "<< endl;
	ofstream file_19;
	file_19.open ("cnode_cpp.txt");
	for (size_t i = 0; i < totnode; ++i) {
		for (size_t j = 0; j < numfam[i]; ++j){
			cnode = nodefam[pointfam[i]+j];
			file_19 <<i<<"   "<<j<<"   "<<cnode<<endl;
		}
	}
	file_19.close();
	
	//Time integration
	ofstream file_35;
	file_35.open ("steady_checkbt_cpp.txt");
	
	for (size_t tt = 0; tt < nt; ++tt) {
		cout<<"tt = "<<tt<<endl;
		for (size_t i = 0; i < totint; ++i) { // loop until totint
			pforce[i][0] = 0.0;
			pforce[i][1]= 0.0;
			pforce[i][2] = 0.0;
			for (size_t j = 0; j < numfam[i]; ++j){
					cnode = nodefam[pointfam[i]+j];
					idist = sqrt(pow((coord[cnode][0]-coord[i][0]),2) + pow((coord[cnode][1]-coord[i][1]),2) + \
						pow((coord[cnode][2] - coord[i][2]),2));
					nlength = sqrt(pow((coord[cnode][0] + disp[cnode][0]-coord[i][0] - disp[i][0]),2) + \
						pow((coord[cnode][1] + disp[cnode][1] - coord[i][1] - disp[i][1]),2) + \
						pow((coord[cnode][2] + disp[cnode][2] - coord[i][2] - disp[i][2]),2));
					
					//Volume correction
					if (idist<(delta-radij)){
						fac = 1.0;
					}
					else if (idist<(delta+radij)){
						fac = (delta+radij-idist)/(2.0*radij);
					}
					else{
						fac = 0.0;
					}
					//Determination of the surface correction between two material points
					 if (abs(coord[cnode][2]-coord[i][2]) <= 1.0e-10) {
						if (abs(coord[cnode][1]-coord[i][1]) <= 1.0e-10) {
							theta = 0.0;
						}
						else if (abs(coord[cnode][0] - coord[i][0]) <= 1.0e-10) {
							theta = 90.0 * M_PI / 180.0;
						}
						else {
							theta = atan(abs(coord[cnode][1]-coord[i][1])/abs(coord[cnode][0] - coord[i][0]));
						}
						phi = 90.0 * M_PI / 180.0;

						scx = (fncst[i][0] + fncst[cnode][0] ) / 2.0;
						scy = (fncst[i][1] + fncst[cnode][1] ) / 2.0;
						scz = (fncst[i][2] + fncst[cnode][2] ) / 2.0;
						scr = 1.0/(pow((cos(theta)*sin(phi)),2)/pow(scx,2)+\
							pow((sin(theta)*sin(phi)),2)/pow(scy,2)\
							+pow(cos(phi),2)/pow(scz,2));
						scr = sqrt(scr);
					 }
					else if ((abs(coord[cnode][0]-coord[i][0]) <= 1.0e-10) && (abs(coord[cnode][1]-coord[i][1]) <= 1.0e-10)){
						scz = (fncst[i][2] + fncst[cnode][2]) / 2.0;
						scr = scz;
					 }
					else {
						theta = atan(abs(coord[cnode][1] - coord[i][1]) / abs(coord[cnode][0]-coord[i][0]));
						phi = acos(abs(coord[cnode][2]-coord[i][2])/idist);

						scx = (fncst[i][0] + fncst[cnode][0] ) / 2.0;
						scy = (fncst[i][1] + fncst[cnode][1] ) / 2.0;
						scz = (fncst[i][2] + fncst[cnode][2] ) / 2.0;
						scr = 1.0/(pow((cos(theta)*sin(phi)),2)/pow(scx,2)+\
							pow((sin(theta)*sin(phi)),2)/pow(scy,2)\
							+pow(cos(phi),2)/pow(scz,2));
						scr = sqrt(scr);
					 }       
				   
					//Calculation of the peridynamic force in x, y and z directions 
					//acting on a material point i due to a material point j
					dforce1 = bc*((nlength - idist) / idist - (alpha * dtemp)) * vol*scr * fac * (coord[cnode][0] + disp[cnode][0] - coord[i][0] - disp[i][0]) / nlength;
					dforce2 = bc*((nlength - idist) / idist - (alpha * dtemp)) * vol*scr * fac * (coord[cnode][1] + disp[cnode][1] - coord[i][1] - disp[i][1]) / nlength;
					dforce3 = bc*((nlength - idist) / idist - (alpha * dtemp)) * vol*scr * fac * (coord[cnode][2] + disp[cnode][2] - coord[i][2] - disp[i][2]) / nlength;
					
					pforce[i][0] += dforce1;     
					pforce[i][1] += dforce2;
					pforce[i][2] += dforce3;
			}
		}
	
	ofstream file_20;
	file_20.open ("pforce_cpp.txt");
	for (size_t i = 0; i < totnode; ++i) {
		file_20 <<pforce[i][0]<<"   "<<pforce[i][1]<<"   "<<pforce[i][2]<<endl;
	} 
	file_20.close();
	
	//Adaptive dynamic relaxation (1)
    /*
	for (size_t i = 0; i < totnode; ++i){
        if (velhalfold[i][0]!=0.0){
            cn1 -=disp[i][0]*disp[i][0]* (pforce[i][0] / massvec[i][0]-pforceold[i][0]/ massvec[i][0]) / (dt * velhalfold[i][0]);
        }
        if (velhalfold[i][1]!=0.0) {
            cn1 -=disp[i][1] * disp[i][1]* (pforce[i][1]/massvec[i][1]-pforceold[i][1] / massvec[i][1]) / (dt * velhalfold[i][1]);
        }
        if (velhalfold[i][2]!=0.0) {
            cn1 -= disp[i][2]*disp[i][2]*(pforce[i][2] / massvec[i][2]-pforceold[i][2] / massvec[i][2]) / (dt * velhalfold[i][2]);
        }
		cn2 += disp[i][0] * disp[i][0];
		cn2 += disp[i][1] * disp[i][1];
		cn2 += disp[i][2]* disp[i][2];
    }
	
	if (cn2!= 0.0) {
        if ((cn1 / cn2) > 0.0){
            cn = 2.0*sqrt(cn1 / cn2);
		}
        else{
            cn = 0.0;
		}
    }
    else{
        cn = 0.0;
    }
	
	if (cn > 2.0){
		cn = 1.9;
	}
	for (size_t i = 0; i < totint; ++i){ //loop until totint
        //Integrate acceleration over time
		if (tt==1) {
            velhalf[i][0] = 1.0* dt / massvec[i][0] * (pforce[i][0]+ bforce[i][0] ) / 2.0;
            velhalf[i][1] = 1.0* dt / massvec[i][1]* (pforce[i][1] + bforce[i][1]) / 2.0;
            velhalf[i][2] = 1.0* dt / massvec[i][2] * (pforce[i][2] + bforce[i][2]) / 2.0;
		}
        else{
            velhalf[i][0] = ((2.0- cn * dt) * velhalfold[i][0] + 2.0 * dt \
				/ massvec[i][0] *(pforce[i][0] + bforce[i][0])) / (2.0+ cn * dt);
            velhalf[i][1] = ((2.0- cn * dt) * velhalfold[i][1] + 2.0 * dt \
				/ massvec[i][1] *(pforce[i][1] + bforce[i][1])) / (2.0+ cn * dt);
            velhalf[i][2] = ((2.0- cn * dt) * velhalfold[i][2] + 2.0 * dt \
				/ massvec[i][2] * (pforce[i][2] + bforce[i][2])) / (2.0 + cn * dt);
		}
   
        vel[i][0] = 0.5 * (velhalfold[i][0] + velhalf[i][0]);
        vel[i][1] = 0.5 * (velhalfold[i][1] + velhalf[i][1]);
        vel[i][2] = 0.5 * (velhalfold[i][2] + velhalf[i][2]);
        disp[i][0] += velhalf[i][0] * dt;
        disp[i][1] += velhalf[i][1] * dt;
        disp[i][2] += velhalf[i][2] * dt;
        
        velhalfold[i][0] = velhalf[i][0];
        velhalfold[i][1] = velhalf[i][1];
        velhalfold[i][2] = velhalf[i][2];
		pforceold[i][0] = pforce[i][0];
		pforceold[i][1] = pforce[i][1];
		pforceold[i][2] = pforce[i][2];
    }
	
	//Adaptive dynamic relaxation (2)
	
	if (tt==nt) {
		ofstream file_26;
		file_26.open ("coord_disp_pd_ntbt_cpp.txt");//printing results to an output file
		for (size_t i = 0; i < totint; ++i){ //loop until totint			
			//file_26 <<coord[i][0]<<"   "<<coord[i][1]<<"   "<<coord[i][2]<<"   "<<disp[i][0]<<"   "<<disp[i][1]<<"   "<<disp[i][2]<<endl;
			file_26 <<coord[i][0]<<"   "<<coord[i][1]<<"   "<<coord[i][2]<<"   "<<fncst[i][0]<<"   "<<fncst[i][1]<<"   "<<fncst[i][2]<<endl;
		}

		file_26.close();

        ofstream file_27;
		file_27.open ("horizontal_dispsbt_cpp.txt");

		for (size_t i = 0; i < totint; ++i){ //loop until totint
            if ((abs(coord[i][1]-(dx/2.0))<1.0e-8) && (abs(coord[i][2]-(dx/2.0))<1.0e-8)){
                andisp[i][0] = 0.001* coord[i][0];
                andisp[i][1]= -1.0* 0.001* pratio * coord[i][1];
                andisp[i][2] = -1.0* 0.001* pratio * coord[i][2];
				file_27 <<coord[i][0]<<"   "<<coord[i][1]<<"   "<<coord[i][2]<<"   " \
					<<disp[i][0]<<"   "<<disp[i][1]<<"   "<<disp[i][2]<<"   " \
					<<andisp[i][0]<<"   "<<andisp[i][1]<<"   "<<andisp[i][2]<<endl;
			}
		}
		file_27.close();
        
		ofstream file_28;
		file_28.open ("vertical_dispsbt_cpp.txt");

		for (size_t i = 0; i < totint; ++i){ //loop until totint
            if ((abs(coord[i][0]-(length / 2.0+ dx / 2.0))<1.0e-8) && (abs(coord[i][2]-(dx/2.0))<1.0e-8)){
                andisp[i][0] = 0.001* coord[i][0];
                andisp[i][1]= -1.0* 0.001* pratio * coord[i][1];
                andisp[i][2] = -1.0* 0.001* pratio * coord[i][2];
			    file_28 <<coord[i][0]<<"   "<<coord[i][1]<<"   "<<coord[i][2]<<"   " \
					<<disp[i][0]<<"   "<<disp[i][1]<<"   "<<disp[i][2]<<"   " \
					<<andisp[i][0]<<"   "<<andisp[i][1]<<"   "<<andisp[i][2]<<endl;
			}
		}

		file_28.close();
		
		ofstream file_29;
		file_29.open ("transverse_dispsbt_cpp.txt");
		
		for (size_t i = 0; i < totint; ++i){ //loop until totint
            if ((abs(coord[i][0]-(length / 2.0+ dx/2.0))<1.0e-8) && (abs(coord[i][1]-(dx/2.0))<1.0e-8)){
                andisp[i][0] = 0.001* coord[i][0];
                andisp[i][1]= -1.0* 0.001* pratio * coord[i][1];
                andisp[i][2] = -1.0* 0.001* pratio * coord[i][2];
			    file_29 <<coord[i][0]<<"   "<<coord[i][1]<<"   "<<coord[i][2]<<"   " \
					<<disp[i][0]<<"   "<<disp[i][1]<<"   "<<disp[i][2]<<"   " \
					<<andisp[i][0]<<"   "<<andisp[i][1]<<"   "<<andisp[i][2]<<endl;
			}
		}

		file_29.close();
    }
	
	file_35 <<tt<<"   "<<disp[7769][0]<<"   "<<disp[7769][1]<<"   "<< disp[7769][2]<<endl;
	*/
	} // end of time integration
	
	
	file_35.close();
	cout<<"totint= "<<totint <<", totnode = "<<totnode<<endl;
	printf("End of program!");	
	
}