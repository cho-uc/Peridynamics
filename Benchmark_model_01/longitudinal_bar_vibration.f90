program main

implicit none

integer ndivx, totnode, nt, maxfam, nnum, cnode, i, j, tt, nbnd
!ndivx: Number of divisions in x direction - except boundary region
parameter(ndivx = 1000)
!nbnd: Number of divisions in the boundary region
parameter(nbnd = 3)
!totnode: Total number of material points
parameter (totnode = ndivx + nbnd) 
!nt: Total number of time step
parameter(nt = 26000)
!maxfam: Maximum number of material points inside a horizon of a material point
parameter(maxfam = 100)

real *8 length, dx, delta, dens, emod, area, vol, bc 
real *8 sedload1, dt, totime, ctime, idist, fac, radij, nlength, dforce1 
real *8 pi, tmpdx, tmpvol, tmpcx, tmpux, dmgpar1, dmgpar2, theta 
real *8 scr, coordx, cwave
integer nrao, ntotrao

real *8 coord(totnode,1), pforce(totnode,1), bforce(totnode,1), stendens(totnode,1)
real *8 fncst(totnode,1), fncstold(totnode,1), disp(totnode,1), vel(totnode,1)
real *8 acc(totnode,1), massvec(totnode,1)
real *8 pddisp(nt,1), pdtime(nt,1), andisp(nt,1)
integer numfam(totnode,1), pointfam(totnode,1), nodefam(10000000,1)

pi = dacos(-1.0d0)

do i = 1, totnode
    !coord: Material point locations
	coord(i,1) = 0.0d0
    !numfam: Number of family members of each material point
	numfam(i,1) = 0
    !pointfam: index array to find the family members in nodefam array
	pointfam(i,1) = 0
    !pforce: total peridynamic force acting on a material point
	pforce(i,1) = 0.0d0
    !bforce: body load acting on a material point
	bforce(i,1) = 0.0d0
    !stendens: strain energy of a material point
	stendens(i,1) = 0.0d0
    !fncst: surface correction factor of a material point
	fncst(i,1) = 1.0d0
	fncstold(i,1) = 1.0d0 
    !disp: displacement of a material point
	disp(i,1) = 0.0d0
    !vel: velocity of a material point    
	vel(i,1) = 0.0d0
    !acc: acceleration of a material point
	acc(i,1) = 0.0d0
enddo

do i = 1,nt
    !andisp: analytical displacements for results
    andisp(i,1) = 0.0d0
    !pddisp: peridynamic displacements for results
    pddisp(i,1) = 0.0d0
    !pdtime: time array for results
    pdtime(i,1) = 0.0d0
enddo

do i = 1, 1000000
    !nodefam: array containing family members of all material points
	nodefam(i,1) = 0
enddo

!length: Total length of the bar
length = 1.0d0
!dx: Spacing between material points
dx = length / ndivx
!delta: Horizon
delta = 3.015d0 * dx
!dens: Density
dens = 7850.0d0
!emod: Elastic modulus
emod = 200.0d9
!area: Cross-sectional area
area = dx * dx
!vol: Volume of a material point
vol = area * dx
!bc: Bond constant 
bc = 2.0d0 * emod / (area * (delta**2))
!sedload1: strain energy density of a material point for the first loading condition
!based on classical continuum mechanics
sedload1 = 0.5d0 * emod * 1.0d-6  
!dt: Time step size
dt = 0.8d0 * dsqrt(2.0d0*dens*dx/(2.0d0*delta*area*bc))
!totime: Total time
totime = nt * dt
!ctime: Current time
ctime = 0.0d0
!idist: Initial distance between two material points
idist = 0.0d0
!fac: Volume correction factor
fac = 0.0d0
!radij: Material point radius
radij = dx / 2.0d0
!nnum: Material point number
nnum = 0
!cnode: Current material point
cnode = 0
!nlength: Current distance between two material points
nlength  = 0.0d0
!dforce1: x component of the PD force between two material points
dforce1 = 0.0d0
!ntotrao: Number of terms in the summation for the analytical displacement calculation
ntotrao = 20
!cwave: Wave speed
cwave = dsqrt(emod / dens)

!Specification of the locations of material points
!Material points of the bar
do i = 1,ndivx
        coordx = (dx / 2.0d0) + (i - 1) * dx
        nnum = nnum + 1
        coord(nnum,1) = coordx
enddo

!Material points of the constrained region
do i = 1,nbnd
        coordx = (-1.0d0 / 2.0d0 * dx) - (i - 1) * dx
        nnum = nnum + 1
        coord(nnum,1) = coordx
enddo

!Determination of material points inside the horizon of each material point
do i = 1,totnode
    if (i.eq.1) then 
        pointfam(i,1) = 1
    else
        pointfam(i,1) = pointfam(i-1,1) + numfam(i-1,1)
    endif
    do j = 1,totnode
        idist = dsqrt((coord(j,1) - coord(i,1))**2)
        if (i.ne.j) then
            if(idist.le.delta) then
                numfam(i,1) = numfam(i,1) + 1
                nodefam(pointfam(i,1)+numfam(i,1)-1,1) = j
            endif
        endif
    enddo
enddo

!Determination of surface correction factors 
!Loading 1
do i = 1,totnode
    disp(i,1) = 0.001d0 * coord(i,1)
enddo
   
do i = 1,totnode
    stendens(i,1) = 0.0d0
    do j = 1,numfam(i,1)
        cnode = nodefam(pointfam(i,1)+j-1,1)
        idist = dsqrt((coord(cnode,1) - coord(i,1))**2)
        nlength = dsqrt((coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1))**2)
        if (idist.le.delta-radij) then
            fac = 1.0d0
        elseif (idist.le.delta+radij) then
            fac = (delta+radij-idist)/(2.0d0*radij)
        else
            fac = 0.0d0
        endif 

        stendens(i,1) = stendens(i,1) + 0.5d0 * 0.5d0 * bc * ((nlength - idist) / idist)**2 * idist * vol * fac  
    enddo
    !Calculation of surface correction factor in x direction 
    !by finding the ratio of the analytical strain energy density value
    !to the strain energy density value obtained from PD Theory
    fncst(i,1) = sedload1 / stendens(i,1)
enddo
       
!Initialization of displacements and velocities
do i = 1,totnode
    vel(i,1) = 0.0d0
    disp(i,1) = 0.0d0      
enddo

!Initial condition 
do i = 1,ndivx
    vel(i,1) = 0.0d0
    disp(i,1) = 0.001d0 * coord(i,1)  !epsilon*H*(delta_t-t)
enddo

open(14,file = 'disp_init.txt')
do i = 1,totnode
	write(14,222) disp(i,1)
enddo
close(14)

!Boundary condition - Zero displacement at x = 0
do i = (ndivx+1),totnode
    vel(i,1) = 0.0d0 
    disp(i,1) = 0.0d0
enddo

!Time integration
do tt = 1,nt
    if (mod(tt,100).eq.0) then
		write(*,*) 'tt = ', tt
	endif	
	
    ctime = tt * dt

    do i = 1,ndivx
        pforce(i,1) = 0.0d0
        do j = 1,numfam(i,1)            
                cnode = nodefam(pointfam(i,1)+j-1,1)
                idist = dsqrt((coord(cnode,1) - coord(i,1))**2)
                nlength = dsqrt((coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1))**2)
                
                !Volume correction
                if (idist.le.delta-radij) then
                    fac = 1.0d0
                elseif (idist.le.delta+radij) then
                    fac = (delta+radij-idist)/(2.0d0*radij)
                else
                    fac = 0.0d0
                endif
                
                !Determination of the surface correction between two material points
                scr = (fncst(i,1) + fncst(cnode,1)) / 2.0d0
               
                !Calculation of the peridynamic force in x direction 
                !acting on a material point i due to a material point j
                dforce1=bc*(nlength-idist)/idist*vol*scr*fac*(coord(cnode,1)+disp(cnode,1)-coord(i,1)-disp(i,1))/nlength
                
                pforce(i,1) = pforce(i,1) + dforce1                                                    				          
        enddo
    enddo
    
    do i = 1, ndivx
        !Calculate the acceleration of material point i
        acc(i,1) = (pforce(i,1) + bforce(i,1)) / dens
        !Calculate the velocity of material point i
        !by integrating the acceleration of material point i
        vel(i,1) = vel(i,1) + acc(i,1) * dt
        !Calculate the displacement of material point i
        !by integrating the velocity of material point i
        disp(i,1) = disp(i,1) + vel(i,1) * dt       
    enddo
    
    !Store the displacement and time information for the material point at the center
    !of the bar for results
    pddisp(tt,1) = disp(500,1)
    pdtime(tt,1) = ctime
    !Calculate the analytical displacement solution of the material point at the center   
    !of the bar
    do nrao = 0, ntotrao
       andisp(tt,1)=andisp(tt,1)+((-1.0d0)**(nrao))/((2.0d0 * nrao + 1.0d0)**2)&
	   &*dsin((2.0d0 * nrao + 1.0d0)*pi*coord(500,1)/2.0d0)*dcos((2.0d0 * nrao + 1.0d0)*pi*cwave*ctime/2.0d0)
    enddo
    andisp(tt,1) = 8.0d0 * 0.001d0 * 1.0d0 / (pi**2) * andisp(tt,1)

enddo !end of time integration

!printing results to an output file
open(26,file = 'coord_disp_pd_nt.txt')

do i = 1, nt
    write(26,111) pdtime(i,1), pddisp(i,1), andisp(i,1)
enddo

close(26)

open(15,file = 'bforce.txt')
do i = 1,totnode
	write(15,222) bforce(i,1)
enddo
close(15)

111 format(e12.5,3x,e12.5,3x,e12.5)
222 format(e12.5)

end program main