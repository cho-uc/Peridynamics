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
parameter(nt = 10000)
!parameter(nt = 10)
!maxfam: Maximum number of material points inside a horizon of a material point
parameter(maxfam = 100)

real *8 length, dx, delta, dens, emod, area, vol, bc 
real *8 sedload1, dt, totime, ctime, idist, fac, radij, nlength, dforce1 
real *8 pi, tmpdx, tmpvol, tmpcx, tmpux, dmgpar1, dmgpar2, theta 
real *8 scr, coordx, cn, cn1, cn2, appres

real *8 coord(totnode,1), pforce(totnode,1), pforceold(totnode,1), bforce(totnode,1), stendens(totnode,1)
real *8 fncst(totnode,1), fncstold(totnode,1), disp(totnode,1), vel(totnode,1), &
& velhalfold(totnode,1), velhalf(totnode,1), andisp(totnode,1)
real *8 acc(totnode,1), massvec(totnode,1)
integer numfam(totnode,1), pointfam(totnode,1), nodefam(10000000,1), remainder

pi = dacos(-1.0d0)

!coord: Material point locations
do i = 1, totnode 
    !coord: Material point locations
	coord(i,1) = 0.0d0
    !numfam: Number of family members of each material point
	numfam(i,1) = 0
    !pointfam: index array to find the family members in nodefam array
	pointfam(i,1) = 0
    !pforce: total peridynamic force acting on a material point
	pforce(i,1) = 0.0d0
    !pforceold: total peridynamic force acting on a material point in the previous time step
	pforceold(i,1) = 0.0d0
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
	velhalfold(i,1) = 0.0d0
	velhalf(i,1) = 0.0d0
    !acc: acceleration of a material point
	acc(i,1) = 0.0d0
    !massvec: massvector for adaptive dynamic relaxation
	massvec(i,1) = 0.0d0
    !andisp: analytical displacements for results
    andisp(i,1) = 0.0d0
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
!dt: Time interval
dt = 1.0d0
!totime: Total time
totime = nt * dt
!ctime: Current time
ctime = 0.0d0
!idist: Initial distance
idist = 0.0d0
!fac: Volume correction factor
fac = 0.0d0
!radij: Material point radius
radij = dx / 2.0d0
!nnum: Material point number
nnum = 0
!cnode: Current material point
cnode = 0
!Length of deformed bond
nlength  = 0.0d0
!dforce1: x component of the PD force between two material points
dforce1 = 0.0d0
!appres: Applied pressure
appres = 200.0d6

!Specification of the locations of material points
!Material points of the bar
do i = 1,ndivx
        coordx = (dx / 2.0d0) + (i - 1) * dx
        nnum = nnum + 1
        coord(nnum,1) = coordx
enddo

write(*,*) 'nnum after material update =', nnum

!Material points of the constrained region
do i = 1,nbnd
        coordx = (-1.0d0 / 2.0d0 * dx) - (i - 1) * dx
        nnum = nnum + 1
        coord(nnum,1) = coordx
enddo

open(14,file = 'coord.txt')
do i = 1,totnode
	write(14,111) coord(i,1), coord(i,2), coord(i,3)
enddo
close(14)

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

!Stable mass vector computation
do i = 1,totnode
   !5 is a safety factor
   massvec(i,1) = 0.25d0 * dt * dt * (2.0d0 * area * delta) * bc / dx * 5.0d0
enddo

!Applied loading
bforce(ndivx,1) = appres / (dx)

open(15,file = 'bforce.txt')
do i = 1,totnode
	write(15,333) bforce(i,1)
enddo
close(15)

!Boundary condition - Constrained region
do i = (ndivx+1),totnode
    vel(i,1) = 0.0d0 
    disp(i,1) = 0.0d0
enddo

open(35,file='center_node.txt')


!Time integration
do tt = 1,nt
	
	if (mod(tt,100).eq.0) then
		write(*,*) 'tt = ', tt
	endif
	
    do i = 1,totnode
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
                dforce1 = bc * (nlength - idist) / idist * vol * scr * fac *&
					&(coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1)) / nlength             
                
                pforce(i,1) = pforce(i,1) + dforce1                                                    				          
        enddo
    enddo
    
    !Adaptive dynamic relaxation
    cn = 0.0d0
	cn1 = 0.0d0
	cn2 = 0.0d0
	do i = 1,ndivx
        if (velhalfold(i,1).ne.0.0d0) then
            cn1 = cn1 - disp(i,1) * disp(i,1) * (pforce(i,1) / massvec(i,1) - &
			& pforceold(i,1) / massvec(i,1)) / (dt * velhalfold(i,1))
        endif
		cn2 = cn2 + disp(i,1) * disp(i,1)
    enddo

	if (cn2.ne.0.0d0) then
        if ((cn1 / cn2) > 0.0d0) then 
            cn = 2.0d0 * dsqrt(cn1 / cn2)
        else
            cn = 0.0d0
        endif
    else
        cn = 0.0d0
    endif

	if (cn > 2.0d0) then
		cn = 1.9d0
	endif

	do i = 1,ndivx
        ! Integrate acceleration over time. 
		if (tt.eq.1) then
            velhalf(i,1) = 1.0d0 * dt / massvec(i,1) * (pforce(i,1) + bforce(i,1)) / 2.0d0			
        else	
            velhalf(i,1) = ((2.0d0 - cn * dt) * velhalfold(i,1) + 2.0d0 * dt / massvec(i,1) * &
				&(pforce(i,1) + bforce(i,1))) / (2.0d0 + cn * dt)
        endif
   
        vel(i,1) = 0.5d0 * (velhalfold(i,1) + velhalf(i,1))	
        disp(i,1) = disp(i,1) + velhalf(i,1) * dt	

        velhalfold(i,1) = velhalf(i,1)
		pforceold(i,1) = pforce(i,1)
    enddo    
    
    !Adaptive dynamic relaxation
    
	if (tt.eq.nt) then
        !printing results to an output file
		open(26,file = 'coord_disp_pd_nt.txt')

		do i = 1, ndivx
            andisp(i,1) = 0.001d0 * coord(i,1)
			write(26,111) coord(i,1), disp(i,1), andisp(i,1)
		enddo

		close(26)
    endif
    
    write(35,222) tt, disp(500,1) !displacement in the middle of the node

enddo

111 format(e12.5,3x,e12.5,3x,e12.5)
222 format(i10,3x,e12.5)
333 format(e12.5)
    
close(35)

end program main