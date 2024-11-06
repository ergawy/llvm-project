! Tests that if `do concurrent` is indirectly nested in its parent loop, that we
! skip converting the indirectly nested `do concurrent` loop.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=host %s -o - \
! RUN:   | FileCheck %s --check-prefixes=HOST,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %s -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,COMMON

program main
   integer, parameter :: n = 10
   integer, parameter :: m = 20
   integer, parameter :: l = 30
   integer x;
   integer :: a(n, m, l)

   do concurrent(i=1:n)
     do j=1,m
       do concurrent(k=1:l)
         a(i,j,k) = i * j + k
       end do
     end do
   end do
end

! HOST: %[[ORIG_K_ALLOC:.*]] = fir.alloca i32 {bindc_name = "k"}
! HOST: %[[ORIG_K_DECL:.*]]:2 = hlfir.declare %[[ORIG_K_ALLOC]]

! HOST: %[[ORIG_J_ALLOC:.*]] = fir.alloca i32 {bindc_name = "j", {{.*}}}
! HOST: %[[ORIG_J_DECL:.*]]:2 = hlfir.declare %[[ORIG_J_ALLOC]]

! DEVICE: omp.target {{.*}}map_entries(
! DEVICE-SAME:   %{{[^[:space:]]+}} -> {{[^[:space:]]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> {{[^[:space:]]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> {{[^[:space:]]+}},
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[I_ARG:[^[:space:]]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[J_ARG:[^[:space:]]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[K_ARG:[^[:space:]]+]],
! DEVICE-SAME:   %{{[^[:space:]]+}} -> %[[A_ARG:[^[:space:]]+]] : {{.*}}) {

! DEVICE: %[[TARGET_J_DECL:.*]]:2 = hlfir.declare %[[J_ARG]] {uniq_name = "_QFEj"}
! DEVICE: %[[TARGET_K_DECL:.*]]:2 = hlfir.declare %[[K_ARG]] {uniq_name = "_QFEk"}

! DEVICE: omp.teams

! COMMON: omp.parallel {

! DEVICE: omp.distribute

! COMMON: omp.wsloop {
! COMMON: omp.loop_nest ({{[^[:space:]]+}}) {{.*}} {
! COMMON:   fir.do_loop {{.*}} iter_args(%[[J_IV:.*]] = {{.*}}) -> {{.*}} {
! HOST:       fir.store %[[J_IV]] to %[[ORIG_J_DECL]]#1
! DEVICE:     fir.store %[[J_IV]] to %[[TARGET_J_DECL]]#1

! COMMON:     fir.do_loop %[[K_IV:.*]] = {{.*}} {
! COMMON:       %[[K_IV_CONV:.*]] = fir.convert %[[K_IV]] : (index) -> i32
! HOST:         fir.store %[[K_IV_CONV]] to %[[ORIG_K_DECL]]#1
! DEVICE:       fir.store %[[K_IV_CONV]] to %[[TARGET_K_DECL]]#1
! COMMON:     }
! COMMON:   }
! COMMON: omp.yield
! COMMON: }
! COMMON: }
! COMMON: omp.terminator
! COMMON: }
