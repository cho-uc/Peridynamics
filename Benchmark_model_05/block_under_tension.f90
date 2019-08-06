program main

implicit none

integer ndivx, ndivy, ndivz, totnode, nt, maxfam, nnum, cnode, i, j, k, tt, nbnd
!ndivx: Number of divisions in x direction - except boundary region
!parameter(ndivx = 100)
parameter(ndivx = 10)
!ndivy: Number of divisions in y direction - except boundary region
parameter(ndivy = 10)
!ndivz: Number of divisions in z direction - except boundary region
parameter(ndivz = 10)
!nbnd: Number of divisions in the boundary region
parameter(nbnd = 3)
!totnode: Total number of material points
parameter (totnode = (ndivx + nbnd) * ndivy * ndivz) 
!nt: Total number of time step
!parameter(nt = 4000) !change from 4000 to 40
parameter(nt = 40)
!maxfam: Maximum number of material points inside a horizon of a material point
parameter(maxfam = 100)

real *8 length, width, thick, dx, delta, dens, emod, pratio, area, vol, bc 
real *8 sedload1, sedload2, sedload3, dt, totime, ctime, idist, fac, radij, nlength, dforce1, dforce2, dforce3 
real *8 pi, tmpdx, tmpvol, tmpcx, tmpux, tmpcy, tmpuy, tmpcz, tmpuz, dmgpar1, dmgpar2, theta, phi 
real *8 scr, scx, scy, scz, coordx, coordy, coordz, cn, cn1, cn2, appres, alpha, dtemp
integer totint

real *8 coord(totnode,3), pforce(totnode,3), pforceold(totnode,3), bforce(totnode,3), stendens(totnode,3)
real *8 fncst(totnode,3), fncstold(totnode,3), disp(totnode,3), &
& vel(totnode,3), velhalfold(totnode,3), velhalf(totnode,3), andisp(totnode,3)
real *8 acc(totnode,3), massvec(totnode,3)
integer numfam(totnode,1), pointfam(totnode,1), nodefam(10000000,1), alflag(totnode,1)

pi = dacos(-1.0d0)

do i = 1, totnode 
    !coord: Material point locations, 1:x-coord, 2:y-coord, 3:z-coord
	coord(i,1) = 0.0d0
	coord(i,2) = 0.0d0
	coord(i,3) = 0.0d0
    !numfam: Number of family members of each material point
	numfam(i,1) = 0
    !pointfam: index array to find the family members in nodefam array
	pointfam(i,1) = 0
    !pforce: total peridynamic force acting on a material point 
    !1:x-coord, 2:y-coord, 3:z-coord
	pforce(i,1) = 0.0d0
	pforce(i,2) = 0.0d0
	pforce(i,3) = 0.0d0
    !pforceold: total peridynamic force acting on a material point in the previous time step
    !1:x-coord, 2:y-coord, 3:z-coord
	pforceold(i,1) = 0.0d0
	pforceold(i,2) = 0.0d0
	pforceold(i,3) = 0.0d0
    !bforce: body load acting on a material point, 1:x-coord, 2:y-coord, 3:z-coord
	bforce(i,1) = 0.0d0
	bforce(i,2) = 0.0d0
	bforce(i,3) = 0.0d0
    !stendens: strain energy of a material point, 1:loading 1, 2:loading 2, 3:loading 3
	stendens(i,1) = 0.0d0
	stendens(i,2) = 0.0d0
	stendens(i,3) = 0.0d0
    !fncst: surface correction factors of a material point, 1:loading 1, 2:loading 2, 3:loading 3
	fncst(i,1) = 1.0d0
	fncstold(i,1) = 1.0d0  
	fncst(i,2) = 1.0d0
	fncstold(i,2) = 1.0d0  
	fncst(i,3) = 1.0d0
	fncstold(i,3) = 1.0d0  
    !disp: displacement of a material point, 1:x-coord, 2:y-coord, 3:z-coord 
	disp(i,1) = 0.0d0
	disp(i,2) = 0.0d0
	disp(i,3) = 0.0d0
    !vel: velocity of a material point, 1:x-coord, 2:y-coord, 3:z-coord 
	vel(i,1) = 0.0d0
	vel(i,2) = 0.0d0
	vel(i,3) = 0.0d0
	velhalfold(i,1) = 0.0d0
	velhalfold(i,2) = 0.0d0
	velhalfold(i,3) = 0.0d0
	velhalf(i,1) = 0.0d0
	velhalf(i,2) = 0.0d0
	velhalf(i,3) = 0.0d0
    !acc: acceleration of a material point, 1:x-coord, 2:y-coord, 3:z-coord
	acc(i,1) = 0.0d0
	acc(i,2) = 0.0d0
	acc(i,3) = 0.0d0
    !massvec: massvector for adaptive dynamic relaxation, 1:x-coord, 2:y-coord, 3:z-coord
	massvec(i,1) = 0.0d0
	massvec(i,2) = 0.0d0
	massvec(i,3) = 0.0d0
    !andisp: analytical displacements for results, 1:x-coord, 2:y-coord, 3:z-coord
    andisp(i,1) = 0.0d0
    andisp(i,2) = 0.0d0
    andisp(i,3) = 0.0d0
    alflag(i,1) = 0
enddo

do i = 1, 1000000
	nodefam(i,1) = 0
enddo

!length: Total length of the block
length = 1.0d0
!width: Total width of the block
width = 0.1d0
!thick: Total thickness of the block
thick = 0.1d0
!dx: Spacing between material points
dx = length / ndivx
!delta: Horizon
delta = 3.015d0 * dx
!dens: Density
dens = 7850.0d0
!emod: Elastic modulus
emod = 200.0d9
!pratio: Poisson's ratio
pratio = 1.0d0 / 4.0d0
!alpha: Coefficient of thermal expansion
alpha = 23.0d-6
!dtemp: Temperature change
dtemp = 0.0d0
!area: Cross-sectional area
area = dx * dx
!vol: Volume of a material point
vol = area * dx
!bc: Bond constant 
bc = 12.0d0 * emod / (pi * (delta**4))
!sedload1: Strain energy density of a material point for the first loading condition
sedload1 = 0.6d0 * emod * 1.0d-6    
!sedload2: Strain energy density of a material point for the second loading condition
sedload2 = 0.6d0 * emod * 1.0d-6
!sedload3: Strain energy density of a material point for the third loading condition
sedload3 = 0.6d0 * emod * 1.0d-6
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
!dforce2: y component of the PD force between two material points
dforce2 = 0.0d0
!dforce3: z component of the PD force between two material points
dforce3 = 0.0d0
!appres: Applied pressure
appres = 200.0d6

!Specification of the locations of material points
!Material points of the block
do i = 1,ndivz
    do j = 1,ndivy
        do k = 1,ndivx
            coordx = (dx / 2.0d0) + (k - 1) * dx
            coordy = -1.0d0 /2.0d0 * width + (dx / 2.0d0) + (j - 1) * dx
            coordz = -1.0d0 /2.0d0 * thick + (dx / 2.0d0) + (i - 1) * dx
            nnum = nnum + 1
            coord(nnum,1) = coordx
            coord(nnum,2) = coordy
            coord(nnum,3) = coordz
            if (coordx.gt.length-dx) then
                alflag(nnum,1) = 1
            endif
        enddo
    enddo
enddo

totint = nnum



!Material points of the boundary region - left
do i = 1,ndivz
    do j = 1,ndivy
        do k = 1,nbnd
            coordx = - (dx / 2.0d0) - (k - 1) * dx
            coordy = -1.0d0 /2.0d0 * width + (dx / 2.0d0) + (j - 1) * dx
            coordz = -1.0d0 /2.0d0 * thick + (dx / 2.0d0) + (i - 1) * dx
            nnum = nnum + 1
            coord(nnum,1) = coordx
            coord(nnum,2) = coordy
            coord(nnum,3) = coordz
        enddo
    enddo
enddo

open(14,file = 'coord.txt')
do i = 1,totnode
	write(14,555) coord(i,1), coord(i,2), coord(i,3)
enddo
close(14)

!Determination of material points inside the horizon of each material point
do i = 1,totnode
    if (i.eq.1) then 
        pointfam(i,1) = 1
    else
        pointfam(i,1) = pointfam(i-1,1) + numfam(i-1,1) !pointfam(2)=pointfam(1)+numfam(1)
    endif
    do j = 1,totnode
        idist = dsqrt((coord(j,1) - coord(i,1))**2 + (coord(j,2) - coord(i,2))**2 + (coord(j,3) - coord(i,3))**2)
        if (i.ne.j) then
            if(idist.le.delta) then
                numfam(i,1) = numfam(i,1) + 1
                nodefam(pointfam(i,1)+numfam(i,1)-1,1) = j
            endif
        endif
    enddo
enddo

open(15,file = 'pointfam_numfam_nodefam.txt')
do i = 1,totnode+1000
	write(15,444) pointfam(i,1), numfam(i,1), nodefam(i,1)
enddo
close(15)

!Determination of surface correction factors 
!Loading 1
do i = 1,totnode
    disp(i,1) = 0.001d0 * coord(i,1)
    disp(i,2) = 0.0d0
    disp(i,3) = 0.0d0
enddo
   
do i = 1,totnode
    stendens(i,1) = 0.0d0
    do j = 1,numfam(i,1)
        cnode = nodefam(pointfam(i,1)+j-1,1)
        idist = dsqrt((coord(cnode,1) - coord(i,1))**2 + (coord(cnode,2) - coord(i,2))**2 + &
		& (coord(cnode,3) - coord(i,3))**2)
        nlength = dsqrt((coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1))**2 + &
		& (coord(cnode,2) + disp(cnode,2) - coord(i,2) - disp(i,2))**2 + (coord(cnode,3) + &
		& disp(cnode,3) - coord(i,3) - disp(i,3))**2)
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

!Loading 2
do i = 1,totnode
    disp(i,1) = 0.0d0
    disp(i,2) = 0.001d0 * coord(i,2)
    disp(i,3) = 0.0d0
enddo

do i = 1,totnode
    stendens(i,2) = 0.0d0
    do j = 1,numfam(i,1)
        cnode = nodefam(pointfam(i,1)+j-1,1)
        idist = dsqrt((coord(cnode,1) - coord(i,1))**2 + (coord(cnode,2) - coord(i,2))**2 + &
		&(coord(cnode,3) - coord(i,3))**2)
        nlength = dsqrt((coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1))**2 + &
		&(coord(cnode,2) + disp(cnode,2) - coord(i,2) - disp(i,2))**2 + (coord(cnode,3) + disp(cnode,3) - coord(i,3) - disp(i,3))**2)
        if (idist.le.delta-radij) then
            fac = 1.0d0
        elseif (idist.le.delta+radij) then
            fac = (delta+radij-idist)/(2.0d0*radij)
        else
            fac = 0.0d0
        endif 
         
        stendens(i,2) = stendens(i,2) + 0.5d0 * 0.5d0 * bc * ((nlength - idist) / idist)**2 * idist * vol * fac 
    enddo
    !Calculation of surface correction factor in y direction 
    !by finding the ratio of the analytical strain energy density value
    !to the strain energy density value obtained from PD Theory
    fncst(i,2) = sedload2 / stendens(i,2)
enddo
   
!Loading 3
do i = 1,totnode
    disp(i,1) = 0.0d0
    disp(i,2) = 0.0d0
    disp(i,3) = 0.001d0 * coord(i,3)
enddo

do i = 1,totnode
    stendens(i,3) = 0.0d0
    do j = 1,numfam(i,1)
        cnode = nodefam(pointfam(i,1)+j-1,1)
        idist = dsqrt((coord(cnode,1) - coord(i,1))**2 + (coord(cnode,2) - &
		& coord(i,2))**2 + (coord(cnode,3) - coord(i,3))**2)
        nlength = dsqrt((coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1))**2 + &
		& (coord(cnode,2) + disp(cnode,2) - coord(i,2) - disp(i,2))**2 + (coord(cnode,3) + disp(cnode,3) - coord(i,3) - disp(i,3))**2)
        if (idist.le.delta-radij) then
            fac = 1.0d0
        elseif (idist.le.delta+radij) then
            fac = (delta+radij-idist)/(2.0d0*radij)
        else
            fac = 0.0d0
        endif      
            
        stendens(i,3) = stendens(i,3) + 0.5d0 * 0.5d0 * bc * ((nlength - idist) / idist)**2 * idist * vol * fac  
    enddo
    !Calculation of surface correction factor in z direction 
    !by finding the ratio of the analytical strain energy density value
    !to the strain energy density value obtained from PD Theory
    fncst(i,3) = sedload3 / stendens(i,3)
enddo

open(16,file = 'fncst.txt')
do i = 1,totnode
	write(16,555) fncst(i,1), fncst(i,2), fncst(i,3)
enddo
close(16)
  
!Initialization of displacements and velocities
do i = 1,totnode
    vel(i,1) = 0.0d0
    disp(i,1) = 0.0d0 
    vel(i,2) = 0.0d0
    disp(i,2) = 0.0d0 
    vel(i,3) = 0.0d0
    disp(i,3) = 0.0d0 
enddo

!Stable mass vector computation
do i = 1,totnode
   !5 is a safety factor
   massvec(i,1) = 0.25d0 * dt * dt * ((4.0d0/3.0d0)*pi*(delta)**3) * bc / dx !* 5.0d0
   massvec(i,2) = 0.25d0 * dt * dt * ((4.0d0/3.0d0)*pi*(delta)**3) * bc / dx !* 5.0d0
   massvec(i,3) = 0.25d0 * dt * dt * ((4.0d0/3.0d0)*pi*(delta)**3) * bc / dx !* 5.0d0
enddo


open(17,file = 'massvec.txt')
do i = 1,totnode
	write(17,555) massvec(i,1), massvec(i,2), massvec(i,3)
enddo
close(17)


!Applied loading - Right
do i = 1, totint
    if (alflag(i,1).eq.1) then
        bforce(i,1) = appres / (dx)
    endif
enddo

open(18,file = 'bforce.txt')
do i = 1,totnode
	write(18,555) bforce(i,1), bforce(i,2), bforce(i,3)
enddo
close(18)



open(19,file = 'cnode.txt')
do i = 1,totint
	do j = 1,numfam(i,1)
		cnode = nodefam(pointfam(i,1)+j-1,1)
		write(19,444) i, j, cnode
	enddo
enddo

close(19)

open(35,file='steady_checkbt.txt')



!Time integration
do tt = 1,nt
    write(*,*) 'tt = ', tt

    do i = 1,totint
        pforce(i,1) = 0.0d0
        pforce(i,2) = 0.0d0
        pforce(i,3) = 0.0d0
        do j = 1,numfam(i,1)            
                cnode = nodefam(pointfam(i,1)+j-1,1)
                idist = dsqrt((coord(cnode,1) - coord(i,1))**2 + (coord(cnode,2) - coord(i,2))**2 + &
				& (coord(cnode,3) - coord(i,3))**2)
                nlength = dsqrt((coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1))**2 + &
				& (coord(cnode,2) + disp(cnode,2) - coord(i,2) - disp(i,2))**2 + (coord(cnode,3) + disp(cnode,3) - coord(i,3) - disp(i,3))**2)
                
                !Volume correction
                if (idist.le.delta-radij) then
                    fac = 1.0d0
                elseif (idist.le.delta+radij) then
                    fac = (delta+radij-idist)/(2.0d0*radij)
                else
                    fac = 0.0d0
                endif
                !Determination of the surface correction between two material points
                 if (dabs(coord(cnode,3) - coord(i,3)) <= 1.0d-10) then
                    if (dabs(coord(cnode,2) - coord(i,2)) <= 1.0d-10) then
						theta = 0.0d0
					elseif (dabs(coord(cnode,1) - coord(i,1)) <= 1.0d-10) then
						theta = 90.0d0 * pi / 180.0d0
					else
						theta = datan(dabs(coord(cnode,2) - coord(i,2)) / dabs(coord(cnode,1) - coord(i,1)))
					endif
					phi = 90.0d0 * pi / 180.0d0

					scx = (fncst(i,1) + fncst(cnode,1)) / 2.0d0
					scy = (fncst(i,2) + fncst(cnode,2)) / 2.0d0
					scz = (fncst(i,3) + fncst(cnode,3)) / 2.0d0
					scr = 1.0d0/(((dcos(theta)*dsin(phi))**2/(scx)**2)+&
					&((dsin(theta)*dsin(phi))**2/(scy)**2)+((dcos(phi))**2/(scz)**2))
					scr = dsqrt(scr)
				elseif (dabs(coord(cnode,1) - coord(i,1)) <= 1.0d-10.and.dabs(coord(cnode,2) - coord(i,2)) <= 1.0d-10) then
					scz = (fncst(i,3) + fncst(cnode,3)) / 2.0d0
				    scr = scz
                else
                    theta = datan(dabs(coord(cnode,2) - coord(i,2)) / dabs(coord(cnode,1) - coord(i,1)))
					phi = dacos(dabs(coord(cnode,3) - coord(i,3)) / idist)

					scx = (fncst(i,1) + fncst(cnode,1)) / 2.0d0
					scy = (fncst(i,2) + fncst(cnode,2)) / 2.0d0
					scz = (fncst(i,3) + fncst(cnode,3)) / 2.0d0
					scr = 1.0d0/(((dcos(theta)*dsin(phi))**2/(scx)**2)+((dsin(theta)*dsin(phi))**2/(scy)**2)+((dcos(phi))**2/(scz)**2))
					scr = dsqrt(scr)
                endif                
               
                !Calculation of the peridynamic force in x, y and z directions 
                !acting on a material point i due to a material point j
                dforce1 = bc * ((nlength - idist) / idist - (alpha * dtemp)) * vol * &
				& scr * fac * (coord(cnode,1) + disp(cnode,1) - coord(i,1) - disp(i,1)) / nlength 
                dforce2 = bc * ((nlength - idist) / idist - (alpha * dtemp)) * vol * &
				& scr * fac * (coord(cnode,2) + disp(cnode,2) - coord(i,2) - disp(i,2)) / nlength
                dforce3 = bc * ((nlength - idist) / idist - (alpha * dtemp)) * vol * & 
				&scr * fac * (coord(cnode,3) + disp(cnode,3) - coord(i,3) - disp(i,3)) / nlength
                
                pforce(i,1) = pforce(i,1) + dforce1      
                pforce(i,2) = pforce(i,2) + dforce2  
                pforce(i,3) = pforce(i,3) + dforce3  
        enddo
    enddo
    open(20,file = 'pforce.txt')
	do i = 1,totint
		write(20,555) pforce(i,1), pforce(i,2), pforce(i,3)
	enddo

	close(20)
    
	!Adaptive dynamic relaxation (1)
    cn = 0.0d0
	cn1 = 0.0d0
	cn2 = 0.0d0
	do i = 1,totint
        if (velhalfold(i,1).ne.0.0d0) then
            cn1 = cn1 - disp(i,1) * disp(i,1) * (pforce(i,1) / massvec(i,1) - &
			& pforceold(i,1) / massvec(i,1)) / (dt * velhalfold(i,1))
        endif
        if (velhalfold(i,2).ne.0.0d0) then
            cn1 = cn1 - disp(i,2) * disp(i,2) * (pforce(i,2) / massvec(i,2) - & 
			& pforceold(i,2) / massvec(i,2)) / (dt * velhalfold(i,2))
        endif
        if (velhalfold(i,3).ne.0.0d0) then
            cn1 = cn1 - disp(i,3) * disp(i,3) * (pforce(i,3) / massvec(i,3) - &
			& pforceold(i,3) / massvec(i,3)) / (dt * velhalfold(i,3))
        endif
		cn2 = cn2 + disp(i,1) * disp(i,1)
		cn2 = cn2 + disp(i,2) * disp(i,2)
		cn2 = cn2 + disp(i,3) * disp(i,3)
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

	do i = 1,totint
        ! Integrate acceleration over time. 
		if (tt.eq.1) then
            velhalf(i,1) = 1.0d0 * dt / massvec(i,1) * (pforce(i,1) + bforce(i,1)) / 2.0d0	
            velhalf(i,2) = 1.0d0 * dt / massvec(i,2) * (pforce(i,2) + bforce(i,2)) / 2.0d0
            velhalf(i,3) = 1.0d0 * dt / massvec(i,3) * (pforce(i,3) + bforce(i,3)) / 2.0d0
        else	
            velhalf(i,1) = ((2.0d0 - cn * dt) * velhalfold(i,1) + 2.0d0 * dt / massvec(i,1) * &
			& (pforce(i,1) + bforce(i,1))) / (2.0d0 + cn * dt)
            velhalf(i,2) = ((2.0d0 - cn * dt) * velhalfold(i,2) + 2.0d0 * dt / massvec(i,2) * &
			& (pforce(i,2) + bforce(i,2))) / (2.0d0 + cn * dt)
            velhalf(i,3) = ((2.0d0 - cn * dt) * velhalfold(i,3) + 2.0d0 * dt / massvec(i,3) * &
			& (pforce(i,3) + bforce(i,3))) / (2.0d0 + cn * dt)
        endif
   
        vel(i,1) = 0.5d0 * (velhalfold(i,1) + velhalf(i,1))	
        vel(i,2) = 0.5d0 * (velhalfold(i,2) + velhalf(i,2))	
        vel(i,3) = 0.5d0 * (velhalfold(i,3) + velhalf(i,3))
        disp(i,1) = disp(i,1) + velhalf(i,1) * dt	
        disp(i,2) = disp(i,2) + velhalf(i,2) * dt	
        disp(i,3) = disp(i,3) + velhalf(i,3) * dt
        
        velhalfold(i,1) = velhalf(i,1)
        velhalfold(i,2) = velhalf(i,2)
        velhalfold(i,3) = velhalf(i,3)
		pforceold(i,1) = pforce(i,1)
		pforceold(i,2) = pforce(i,2)
		pforceold(i,3) = pforce(i,3)
    enddo    
       
    !Adaptive dynamic relaxation (2)
    
	if (tt.eq.nt) then
        !printing results to an output file
		open(26,file = 'coord_disp_pd_ntbt.txt')

		do i = 1, totint
			!write(26,111) coord(i,1), coord(i,2), coord(i,3), disp(i,1), disp(i,2), disp(i,3)
            write(26,111) coord(i,1), coord(i,2), coord(i,3), fncst(i,1), fncst(i,2), fncst(i,3)
		enddo

		close(26)

        open(27,file = 'horizontal_dispsbt.txt')

		do i = 1, totint
            if (dabs(coord(i,2) - (dx / 2.0d0)).le.1.0d-8.and.dabs(coord(i,3) - (dx / 2.0d0)).le.1.0d-8) then
                andisp(i,1) = 0.001d0 * coord(i,1)
                andisp(i,2) = -1.0d0 * 0.001d0 * pratio * coord(i,2)
                andisp(i,3) = -1.0d0 * 0.001d0 * pratio * coord(i,3)
			    write(27,222) coord(i,1), coord(i,2), coord(i,3), disp(i,1), disp(i,2), disp(i,3), andisp(i,1), andisp(i,2), andisp(i,3)
            endif
		enddo

		close(27)
        
        open(28,file = 'vertical_dispsbt.txt')

		do i = 1, totint
            if (dabs(coord(i,1) - (length / 2.0d0 + dx / 2.0d0)).le.1.0d-8.and.dabs(coord(i,3) - (dx / 2.0d0)).le.1.0d-8) then
                andisp(i,1) = 0.001d0 * coord(i,1)
                andisp(i,2) = -1.0d0 * 0.001d0 * pratio * coord(i,2)
                andisp(i,3) = -1.0d0 * 0.001d0 * pratio * coord(i,3)
			    write(28,222) coord(i,1), coord(i,2), coord(i,3), disp(i,1), disp(i,2), disp(i,3), andisp(i,1), andisp(i,2), andisp(i,3)
            endif
		enddo

		close(28)

        open(29,file = 'transverse_dispsbt.txt')

		do i = 1, totint
            if (dabs(coord(i,1) - (length / 2.0d0 + dx / 2.0d0)).le.1.0d-8.and.dabs(coord(i,2) - (dx / 2.0d0)).le.1.0d-8) then
                andisp(i,1) = 0.001d0 * coord(i,1)
                andisp(i,2) = -1.0d0 * 0.001d0 * pratio * coord(i,2)
                andisp(i,3) = -1.0d0 * 0.001d0 * pratio * coord(i,3)
			    write(29,222) coord(i,1), coord(i,2), coord(i,3), disp(i,1), disp(i,2), disp(i,3), andisp(i,1), andisp(i,2), andisp(i,3)
            endif
		enddo

		close(29)
    endif
    
    write(35,333) tt, disp(7770,1), disp(7770,2), disp(7770,3)

enddo

write(*,*) 'totint = ', totint, ', totnode = ', totnode

111 format(e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5)
222 format(e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5,3x,e12.5)
333 format(i10,3x,e12.5,3x,e12.5,3x,e12.5) 
444 format(i10,3x,i10,3x,i10)
555 format(e12.5,3x,e12.5,3x,e12.5) 

end program main