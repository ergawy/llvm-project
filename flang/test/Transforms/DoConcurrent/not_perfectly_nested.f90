! Tests that if `do concurrent` is not perfectly nested in its parent loop, that
! we skip converting the not-perfectly nested `do concurrent` loop.


! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=host %s -o - \
! RUN:   | FileCheck %s --check-prefix=HOST

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %s -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,HOST

program main
   integer, parameter :: n = 10
   integer, parameter :: m = 20
   integer, parameter :: l = 30
   integer x;
   integer :: a(n, m, l)

   do concurrent(i=1:n)
     x = 10
     do concurrent(j=1:m, k=1:l)
       a(i,j,k) = i * j + k
     end do
   end do
end

! DEVICE: omp.target
! DEVICE: omp.teams
! DEVICE: omp.distribute

! HOST: omp.parallel {
! HOST: omp.wsloop {
! HOST: omp.loop_nest ({{[^[:space:]]+}}) {{.*}} {
! HOST:   %[[PRIV_J_ALLOC:.*]] = fir.alloca i32 {bindc_name = "j"}
! HOST:   %[[PRIV_J_DECL:.*]]:2 = hlfir.declare %[[PRIV_J_ALLOC]]
! HOST:   fir.do_loop %[[J_IV:.*]] = {{.*}} {
! HOST:     %[[J_IV_CONV:.*]] = fir.convert %[[J_IV]] : (index) -> i32
! HOST:     fir.store %[[J_IV_CONV]] to %[[PRIV_J_DECL]]#1 : !fir.ref<i32>

! HOST:     %[[PRIV_K_ALLOC:.*]] = fir.alloca i32 {bindc_name = "k"}
! HOST:     %[[PRIV_K_DECL:.*]]:2 = hlfir.declare %[[PRIV_K_ALLOC]]
! HOST:     fir.do_loop %[[K_IV:.*]] = {{.*}} {
! HOST:       %[[K_IV_CONV:.*]] = fir.convert %[[K_IV]] : (index) -> i32
! HOST:       fir.store %[[K_IV_CONV]] to %[[PRIV_K_DECL]]#1 : !fir.ref<i32>
! HOST:     }
! HOST:   }
! HOST: omp.yield
! HOST: }
! HOST: omp.terminator
! HOST: }
! HOST: omp.terminator
! HOST: }
