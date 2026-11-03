module user 
  use neko 
  implicit none

contains

  ! Register user defined functions (see user_intf.f90)
  subroutine user_setup(user)
    type(user_t), intent(inout) :: user
    user%initial_conditions => initial_conditions
    user%mesh_setup => pipe_deform
  end subroutine user_setup

  subroutine pipe_deform(msh, time)
    type(mesh_t), intent(inout) :: msh
    type(time_state_t), intent(in) :: time
    msh%apply_deform => create_rough_pipe
  end subroutine pipe_deform

  subroutine create_rough_pipe(msh, x, y, z, lx, ly, lz)
    class(mesh_t) :: msh
    integer, intent(in) :: lx, ly, lz
    real(kind=rp), intent(inout) :: x(lx, ly, lz, msh%nelv)
    real(kind=rp), intent(inout) :: y(lx, ly, lz, msh%nelv)
    real(kind=rp), intent(inout) :: z(lx, ly, lz, msh%nelv)

    integer :: n_points_z, n_points_th, m, n, e, f, g, h, i, j
    real(kind=rp) :: rr, th, llz, llth, phase, k, rm, rr_new, x_new, y_new, u, v, r_crit, rr_new_acc, new_avg_r, halfM, halfN
    real(8), allocatable :: A(:,:), B(:,:)
    llz = 8.*pi
    llth = 2.*pi 
    r_crit = 0.5

    M = extract_M()
    N = extract_N()

    allocate(A(M,N))
    allocate(B(M,N))

    A = extract_A(M,N)
    B = extract_B(M,N)
    
    halfM = ceiling(real(M)/2.0)
    halfN = ceiling(real(N)/2.0)
    do e = 1,msh%nelv
      do f = 1, lz
        
        do g = 1, ly
          
          do h = 1, lx

            !Map cartesian coords to cylndrical coords
            rr = sqrt(x(h,g,f,e)**2 + y(h,g,f,e)**2)
            if (rr .eq. 0) then
              th = 0.0
            else
              th = atan2(y(h,g,f,e),x(h,g,f,e))
            endif

            k = 0.
            rm = 0.
            if ((rr) .gt. r_crit) then
              do i = 1, M
                if (i .le. halfM) then
                  u = i - 1
                else
                  u = i -1 - M
                endif

                do j = 1, N
                  if ( j .le.  halfN) then
                    v = j - 1
                  else 
                    v = j - 1 -N
                  endif
                  phase = 2.0 *pi* (((u* z(h,g,f,e) )/ llz) + ((v*th)/llth))
                  k = A(i,j)*cos(phase) - B(i,j)*sin(phase)
                  rm = rm + k 
                enddo
              enddo
              rr_new = rr+((rm)*((rr-r_crit)/(1.0-r_crit)))
              
              x_new = rr_new*cos(th)
              y_new = rr_new*sin(th)
              x(h,g,f,e) = x_new
              y(h,g,f,e) = y_new
            endif
            
          enddo
        enddo
      enddo
      

    enddo
  end subroutine create_rough_pipe

  ! User defined initial condition
  subroutine initial_conditions(scheme_name, fields)
    character(len=*), intent(in) :: scheme_name
    type(field_list_t), intent(inout) :: fields
    integer :: i
    real(kind=rp) :: uvw(3), x, y, z
    type (field_t), pointer :: u, v, w

    u => fields%get("u")
    v => fields%get("v")
    w => fields%get("w")

    do i = 1, u%size()
       x = u%dof%x(i,1,1,1)
       y = u%dof%y(i,1,1,1)
       z = u%dof%z(i,1,1,1)
       uvw = pipe_ic(x, y, z)

       u%x(i,1,1,1) = uvw(1)
       v%x(i,1,1,1) = uvw(2)
       w%x(i,1,1,1) = uvw(3)
    end do
  end subroutine initial_conditions

  function pipe_ic(x, y, z) result(uvw)
    real(kind=rp) :: x, y, z, uvw(3)
    real(kind=rp) :: pi
    real(kind=rp) :: llr, llt, llz
    real(kind=rp) :: r, t, ur, ut ,uz
    real(kind=rp) :: eps, kt, kz, alpha, beta, ran

    pi = 4_rp * atan(1.0_rp)
    
    llr = 1.0_rp
    llt = pi * 2 * llr
    llz = 8*pi*llr

    

    ! calculate polar coords
    r = sqrt(x**2 + y**2)
    
    if (r .eq. 0) then
      t = 0.0
    else 
      t = atan2(y, x)
    end if


    uvw(1) = 0
    uvw(2) = 0
    uvw(3) = 0

    ! Parabolic laminar inflow
    ur = 0
    ut = 0
    uz = 2.0*(1-r**2)

    ! calculate velocity in polar coords

    ! add large scale perutbation
    eps = 0.1
    kz = 3
    kt = 5
    alpha = kz * 2*pi/llz
    beta = kt * 2 * pi /llt
    uz = uz - eps*alpha*r*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t)
    ur = ur + eps*(r**5-2*r**3+r)*cos(alpha*z+beta*t)
    ut = ut - eps*beta*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t)
    eps = 0.05
    uz = uz - eps*beta*(r**2-1)**2 * sin(alpha*z+beta*t+2)
    ut = ut + eps*alpha*r*(r**2-1)**2 *sin(alpha*z+beta*t+2)

    eps = 0.1
    kz = 4
    kt = -3
    alpha = kz * 2*PI/llz
    beta = kt * 2*PI/llt
    uz = uz - eps*alpha*r*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t)
    ur = ur + eps*(r**5-2*r**3+r)*cos(alpha*z+beta*t)
    ut = ut - eps*beta*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t)
    eps = 0.05
    uz = uz - eps*beta*(r**2-1)**2 * sin(alpha*z+beta*t+2)
    ut = ut + eps*alpha*r*(r**2-1)**2 *sin(alpha*z+beta*t+2)

    ! add small scale perturbation
    eps = 0.05
    kz = 17
    kt = 13
    alpha = kz * 2*pi/llz
    beta = kt*2*pi/llt
    uz = uz - eps*alpha*r*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t+1)
    ur = ur + eps*(r**5-2*r**3+r)*cos(alpha*z+beta*t+1)
    ut = ut - eps*beta*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t+1)

    kz = 11
    kt = -19
    alpha = kz * 2*pi/llz
    beta = kt*2*pi/llt
    uz = uz - eps*alpha*r*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t+1)
    ur = ur + eps*(r**5-2*r**3+r)*cos(alpha*z+beta*t+1)
    ut = ut - eps*beta*(6*r**5-8*r**3+2*r)/(alpha**2*r**2+beta**2)*sin(alpha*z+beta*t+1)  
    
    ! convert polar to cartesian coords
    uvw(1) = ur*cos(t) - ut*sin(t)
    uvw(2) = ur * sin(t) + ut*cos(t)
    uvw(3) = uz

  end function pipe_ic

  function extract_M() result(M)
    integer :: M, ios
    character(len=200) :: line
    
    open(unit = 10, file='SurfaceGenOutput_production.txt', status='old', action='read')
    do
        read(10, '(A)', iostat = ios) line
        if (ios /= 0) exit

        if(index(line, 'M:') > 0) then
            read(10,*) M
        end if
    end do
    close(10)
  end function extract_M
  function extract_N() result(N)
    integer :: N, ios
    character(len=200) :: line
    
    open(unit = 10, file='SurfaceGenOutput_production.txt', status='old', action='read')
    do
        read(10, '(A)', iostat = ios) line
        if (ios /= 0) exit

        if(index(line, 'N:') > 0) then
            read(10,*) N
        end if
    end do
    close(10)
  end function extract_N

  function extract_A(M, N) result(A)
    integer, intent(in) :: M, N
    integer :: i, ios
    real(kind=rp), dimension(M,N) :: A
    character(len=1000) :: line
    character(len=1000) :: buffer

    open(unit = 10, file='SurfaceGenOutput_production.txt', status='old', action='read')

    do 
      read(10, '(A)', iostat=ios) line
      if (ios /= 0) exit

      if (index(line, 'A:') > 0) then
        !read(10,*)
        do i = 1,M
          read(10,*) A(i,:)
        end do
        exit
      end if
    end do
    close(10)
    !print *, 'A: ', A
  end function extract_A

  function extract_B(M, N) result(B)
    integer, intent(in) :: M, N
    integer :: i, ios
    real(kind=rp), dimension(M,N) :: B
    character(len=1000) :: line
    character(len=1000) :: buffer

    open(unit = 10, file='SurfaceGenOutput_production.txt', status='old', action='read')

    do 
      read(10, '(A)', iostat=ios) line
      if (ios /= 0) exit

      if (index(line, 'B:') > 0) then
        do i = 1,M
          read(10,*) B(i,:)
        end do
        exit
      end if
    end do
    close(10)
  end function extract_B
end module user
