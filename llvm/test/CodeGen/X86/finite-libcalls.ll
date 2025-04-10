; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu     | FileCheck %s --check-prefix=GNU
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc  | FileCheck %s --check-prefix=WIN
; RUN: llc < %s -mtriple=x86_64-apple-darwin     | FileCheck %s --check-prefix=MAC
; RUN: llc < %s -mtriple=i686-linux-gnu -global-isel -global-isel-abort=1 | FileCheck %s --check-prefixes=GISEL-X86
; RUN: llc < %s -mtriple=x86_64-linux-gnu -global-isel -global-isel-abort=1 | FileCheck %s --check-prefixes=GISEL-X64

; PR35672 - https://bugs.llvm.org/show_bug.cgi?id=35672
; FIXME: We would not need the function-level attributes if FMF were propagated to DAG nodes for this case.

define float @exp_f32(float %x) #0 {
; GNU-LABEL: exp_f32:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp expf@PLT # TAILCALL
;
; WIN-LABEL: exp_f32:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp expf # TAILCALL
;
; MAC-LABEL: exp_f32:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _expf ## TAILCALL
;
; GISEL-X86-LABEL: exp_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    calll expf
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: exp_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq expf
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf float @llvm.exp.f32(float %x)
  ret float %r
}

define double @exp_f64(double %x) #0 {
; GNU-LABEL: exp_f64:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp exp@PLT # TAILCALL
;
; WIN-LABEL: exp_f64:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp exp # TAILCALL
;
; MAC-LABEL: exp_f64:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _exp ## TAILCALL
;
; GISEL-X86-LABEL: exp_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl 4(%eax), %eax
; GISEL-X86-NEXT:    xorl %edx, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, (%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    calll exp
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: exp_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq exp
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf double @llvm.exp.f64(double %x)
  ret double %r
}

define x86_fp80 @exp_f80(x86_fp80 %x) #0 {
; GNU-LABEL: exp_f80:
; GNU:       # %bb.0:
; GNU-NEXT:    subq $24, %rsp
; GNU-NEXT:    fldt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fstpt (%rsp)
; GNU-NEXT:    callq expl@PLT
; GNU-NEXT:    addq $24, %rsp
; GNU-NEXT:    retq
;
; WIN-LABEL: exp_f80:
; WIN:       # %bb.0:
; WIN-NEXT:    pushq %rsi
; WIN-NEXT:    subq $64, %rsp
; WIN-NEXT:    movq %rcx, %rsi
; WIN-NEXT:    fldt (%rdx)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN-NEXT:    callq expl
; WIN-NEXT:    fldt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt (%rsi)
; WIN-NEXT:    movq %rsi, %rax
; WIN-NEXT:    addq $64, %rsp
; WIN-NEXT:    popq %rsi
; WIN-NEXT:    retq
;
; MAC-LABEL: exp_f80:
; MAC:       ## %bb.0:
; MAC-NEXT:    subq $24, %rsp
; MAC-NEXT:    fldt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fstpt (%rsp)
; MAC-NEXT:    callq _expl
; MAC-NEXT:    addq $24, %rsp
; MAC-NEXT:    retq
;
; GISEL-X86-LABEL: exp_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    calll expl
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: exp_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq expl
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf x86_fp80 @llvm.exp.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

define float @exp2_f32(float %x) #0 {
; GNU-LABEL: exp2_f32:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp exp2f@PLT # TAILCALL
;
; WIN-LABEL: exp2_f32:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp exp2f # TAILCALL
;
; MAC-LABEL: exp2_f32:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _exp2f ## TAILCALL
;
; GISEL-X86-LABEL: exp2_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    calll exp2f
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: exp2_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq exp2f
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf float @llvm.exp2.f32(float %x)
  ret float %r
}

define double @exp2_f64(double %x) #0 {
; GNU-LABEL: exp2_f64:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp exp2@PLT # TAILCALL
;
; WIN-LABEL: exp2_f64:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp exp2 # TAILCALL
;
; MAC-LABEL: exp2_f64:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _exp2 ## TAILCALL
;
; GISEL-X86-LABEL: exp2_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl 4(%eax), %eax
; GISEL-X86-NEXT:    xorl %edx, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, (%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    calll exp2
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: exp2_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq exp2
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf double @llvm.exp2.f64(double %x)
  ret double %r
}

define x86_fp80 @exp2_f80(x86_fp80 %x) #0 {
; GNU-LABEL: exp2_f80:
; GNU:       # %bb.0:
; GNU-NEXT:    subq $24, %rsp
; GNU-NEXT:    fldt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fstpt (%rsp)
; GNU-NEXT:    callq exp2l@PLT
; GNU-NEXT:    addq $24, %rsp
; GNU-NEXT:    retq
;
; WIN-LABEL: exp2_f80:
; WIN:       # %bb.0:
; WIN-NEXT:    pushq %rsi
; WIN-NEXT:    subq $64, %rsp
; WIN-NEXT:    movq %rcx, %rsi
; WIN-NEXT:    fldt (%rdx)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN-NEXT:    callq exp2l
; WIN-NEXT:    fldt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt (%rsi)
; WIN-NEXT:    movq %rsi, %rax
; WIN-NEXT:    addq $64, %rsp
; WIN-NEXT:    popq %rsi
; WIN-NEXT:    retq
;
; MAC-LABEL: exp2_f80:
; MAC:       ## %bb.0:
; MAC-NEXT:    subq $24, %rsp
; MAC-NEXT:    fldt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fstpt (%rsp)
; MAC-NEXT:    callq _exp2l
; MAC-NEXT:    addq $24, %rsp
; MAC-NEXT:    retq
;
; GISEL-X86-LABEL: exp2_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    calll exp2l
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: exp2_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq exp2l
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf x86_fp80 @llvm.exp2.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

define float @log_f32(float %x) #0 {
; GNU-LABEL: log_f32:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp logf@PLT # TAILCALL
;
; WIN-LABEL: log_f32:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp logf # TAILCALL
;
; MAC-LABEL: log_f32:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _logf ## TAILCALL
;
; GISEL-X86-LABEL: log_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    calll logf
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq logf
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf float @llvm.log.f32(float %x)
  ret float %r
}

define double @log_f64(double %x) #0 {
; GNU-LABEL: log_f64:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp log@PLT # TAILCALL
;
; WIN-LABEL: log_f64:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp log # TAILCALL
;
; MAC-LABEL: log_f64:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _log ## TAILCALL
;
; GISEL-X86-LABEL: log_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl 4(%eax), %eax
; GISEL-X86-NEXT:    xorl %edx, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, (%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    calll log
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq log
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf double @llvm.log.f64(double %x)
  ret double %r
}

define x86_fp80 @log_f80(x86_fp80 %x) #0 {
; GNU-LABEL: log_f80:
; GNU:       # %bb.0:
; GNU-NEXT:    subq $24, %rsp
; GNU-NEXT:    fldt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fstpt (%rsp)
; GNU-NEXT:    callq logl@PLT
; GNU-NEXT:    addq $24, %rsp
; GNU-NEXT:    retq
;
; WIN-LABEL: log_f80:
; WIN:       # %bb.0:
; WIN-NEXT:    pushq %rsi
; WIN-NEXT:    subq $64, %rsp
; WIN-NEXT:    movq %rcx, %rsi
; WIN-NEXT:    fldt (%rdx)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN-NEXT:    callq logl
; WIN-NEXT:    fldt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt (%rsi)
; WIN-NEXT:    movq %rsi, %rax
; WIN-NEXT:    addq $64, %rsp
; WIN-NEXT:    popq %rsi
; WIN-NEXT:    retq
;
; MAC-LABEL: log_f80:
; MAC:       ## %bb.0:
; MAC-NEXT:    subq $24, %rsp
; MAC-NEXT:    fldt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fstpt (%rsp)
; MAC-NEXT:    callq _logl
; MAC-NEXT:    addq $24, %rsp
; MAC-NEXT:    retq
;
; GISEL-X86-LABEL: log_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    calll logl
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq logl
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf x86_fp80 @llvm.log.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

define float @log2_f32(float %x) #0 {
; GNU-LABEL: log2_f32:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp log2f@PLT # TAILCALL
;
; WIN-LABEL: log2_f32:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp log2f # TAILCALL
;
; MAC-LABEL: log2_f32:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _log2f ## TAILCALL
;
; GISEL-X86-LABEL: log2_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    calll log2f
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log2_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq log2f
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf float @llvm.log2.f32(float %x)
  ret float %r
}

define double @log2_f64(double %x) #0 {
; GNU-LABEL: log2_f64:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp log2@PLT # TAILCALL
;
; WIN-LABEL: log2_f64:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp log2 # TAILCALL
;
; MAC-LABEL: log2_f64:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _log2 ## TAILCALL
;
; GISEL-X86-LABEL: log2_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl 4(%eax), %eax
; GISEL-X86-NEXT:    xorl %edx, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, (%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    calll log2
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log2_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq log2
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf double @llvm.log2.f64(double %x)
  ret double %r
}

define x86_fp80 @log2_f80(x86_fp80 %x) #0 {
; GNU-LABEL: log2_f80:
; GNU:       # %bb.0:
; GNU-NEXT:    subq $24, %rsp
; GNU-NEXT:    fldt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fstpt (%rsp)
; GNU-NEXT:    callq log2l@PLT
; GNU-NEXT:    addq $24, %rsp
; GNU-NEXT:    retq
;
; WIN-LABEL: log2_f80:
; WIN:       # %bb.0:
; WIN-NEXT:    pushq %rsi
; WIN-NEXT:    subq $64, %rsp
; WIN-NEXT:    movq %rcx, %rsi
; WIN-NEXT:    fldt (%rdx)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN-NEXT:    callq log2l
; WIN-NEXT:    fldt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt (%rsi)
; WIN-NEXT:    movq %rsi, %rax
; WIN-NEXT:    addq $64, %rsp
; WIN-NEXT:    popq %rsi
; WIN-NEXT:    retq
;
; MAC-LABEL: log2_f80:
; MAC:       ## %bb.0:
; MAC-NEXT:    subq $24, %rsp
; MAC-NEXT:    fldt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fstpt (%rsp)
; MAC-NEXT:    callq _log2l
; MAC-NEXT:    addq $24, %rsp
; MAC-NEXT:    retq
;
; GISEL-X86-LABEL: log2_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    calll log2l
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log2_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq log2l
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf x86_fp80 @llvm.log2.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

define float @log10_f32(float %x) #0 {
; GNU-LABEL: log10_f32:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp log10f@PLT # TAILCALL
;
; WIN-LABEL: log10_f32:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp log10f # TAILCALL
;
; MAC-LABEL: log10_f32:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _log10f ## TAILCALL
;
; GISEL-X86-LABEL: log10_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    calll log10f
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log10_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq log10f
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf float @llvm.log10.f32(float %x)
  ret float %r
}

define double @log10_f64(double %x) #0 {
; GNU-LABEL: log10_f64:
; GNU:       # %bb.0:
; GNU-NEXT:    jmp log10@PLT # TAILCALL
;
; WIN-LABEL: log10_f64:
; WIN:       # %bb.0:
; WIN-NEXT:    jmp log10 # TAILCALL
;
; MAC-LABEL: log10_f64:
; MAC:       ## %bb.0:
; MAC-NEXT:    jmp _log10 ## TAILCALL
;
; GISEL-X86-LABEL: log10_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl 4(%eax), %eax
; GISEL-X86-NEXT:    xorl %edx, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, (%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    calll log10
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log10_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    callq log10
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf double @llvm.log10.f64(double %x)
  ret double %r
}

define x86_fp80 @log10_f80(x86_fp80 %x) #0 {
; GNU-LABEL: log10_f80:
; GNU:       # %bb.0:
; GNU-NEXT:    subq $24, %rsp
; GNU-NEXT:    fldt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fstpt (%rsp)
; GNU-NEXT:    callq log10l@PLT
; GNU-NEXT:    addq $24, %rsp
; GNU-NEXT:    retq
;
; WIN-LABEL: log10_f80:
; WIN:       # %bb.0:
; WIN-NEXT:    pushq %rsi
; WIN-NEXT:    subq $64, %rsp
; WIN-NEXT:    movq %rcx, %rsi
; WIN-NEXT:    fldt (%rdx)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN-NEXT:    callq log10l
; WIN-NEXT:    fldt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt (%rsi)
; WIN-NEXT:    movq %rsi, %rax
; WIN-NEXT:    addq $64, %rsp
; WIN-NEXT:    popq %rsi
; WIN-NEXT:    retq
;
; MAC-LABEL: log10_f80:
; MAC:       ## %bb.0:
; MAC-NEXT:    subq $24, %rsp
; MAC-NEXT:    fldt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fstpt (%rsp)
; MAC-NEXT:    callq _log10l
; MAC-NEXT:    addq $24, %rsp
; MAC-NEXT:    retq
;
; GISEL-X86-LABEL: log10_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    calll log10l
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: log10_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq log10l
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf x86_fp80 @llvm.log10.f80(x86_fp80 %x)
  ret x86_fp80 %r
}

define float @pow_f32(float %x) #0 {
; GNU-LABEL: pow_f32:
; GNU:       # %bb.0:
; GNU-NEXT:    movaps %xmm0, %xmm1
; GNU-NEXT:    jmp powf@PLT # TAILCALL
;
; WIN-LABEL: pow_f32:
; WIN:       # %bb.0:
; WIN-NEXT:    movaps %xmm0, %xmm1
; WIN-NEXT:    jmp powf # TAILCALL
;
; MAC-LABEL: pow_f32:
; MAC:       ## %bb.0:
; MAC-NEXT:    movaps %xmm0, %xmm1
; MAC-NEXT:    jmp _powf ## TAILCALL
;
; GISEL-X86-LABEL: pow_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    calll powf
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: pow_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    movaps %xmm0, %xmm1
; GISEL-X64-NEXT:    callq powf
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf float @llvm.pow.f32(float %x, float %x)
  ret float %r
}

define double @pow_f64(double %x) #0 {
; GNU-LABEL: pow_f64:
; GNU:       # %bb.0:
; GNU-NEXT:    movaps %xmm0, %xmm1
; GNU-NEXT:    jmp pow@PLT # TAILCALL
;
; WIN-LABEL: pow_f64:
; WIN:       # %bb.0:
; WIN-NEXT:    movaps %xmm0, %xmm1
; WIN-NEXT:    jmp pow # TAILCALL
;
; MAC-LABEL: pow_f64:
; MAC:       ## %bb.0:
; MAC-NEXT:    movaps %xmm0, %xmm1
; MAC-NEXT:    jmp _pow ## TAILCALL
;
; GISEL-X86-LABEL: pow_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $28, %esp
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    movl 4(%eax), %eax
; GISEL-X86-NEXT:    xorl %edx, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, (%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    movl $8, %edx
; GISEL-X86-NEXT:    addl %esp, %edx
; GISEL-X86-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    movl %eax, 4(%edx)
; GISEL-X86-NEXT:    calll pow
; GISEL-X86-NEXT:    addl $28, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: pow_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    movaps %xmm0, %xmm1
; GISEL-X64-NEXT:    callq pow
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf double @llvm.pow.f64(double %x, double %x)
  ret double %r
}

define x86_fp80 @pow_f80(x86_fp80 %x) #0 {
; GNU-LABEL: pow_f80:
; GNU:       # %bb.0:
; GNU-NEXT:    subq $40, %rsp
; GNU-NEXT:    fldt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fld %st(0)
; GNU-NEXT:    fstpt {{[0-9]+}}(%rsp)
; GNU-NEXT:    fstpt (%rsp)
; GNU-NEXT:    callq powl@PLT
; GNU-NEXT:    addq $40, %rsp
; GNU-NEXT:    retq
;
; WIN-LABEL: pow_f80:
; WIN:       # %bb.0:
; WIN-NEXT:    pushq %rsi
; WIN-NEXT:    subq $80, %rsp
; WIN-NEXT:    movq %rcx, %rsi
; WIN-NEXT:    fldt (%rdx)
; WIN-NEXT:    fld %st(0)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt {{[0-9]+}}(%rsp)
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rcx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %rdx
; WIN-NEXT:    leaq {{[0-9]+}}(%rsp), %r8
; WIN-NEXT:    callq powl
; WIN-NEXT:    fldt {{[0-9]+}}(%rsp)
; WIN-NEXT:    fstpt (%rsi)
; WIN-NEXT:    movq %rsi, %rax
; WIN-NEXT:    addq $80, %rsp
; WIN-NEXT:    popq %rsi
; WIN-NEXT:    retq
;
; MAC-LABEL: pow_f80:
; MAC:       ## %bb.0:
; MAC-NEXT:    subq $40, %rsp
; MAC-NEXT:    fldt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fld %st(0)
; MAC-NEXT:    fstpt {{[0-9]+}}(%rsp)
; MAC-NEXT:    fstpt (%rsp)
; MAC-NEXT:    callq _powl
; MAC-NEXT:    addq $40, %rsp
; MAC-NEXT:    retq
;
; GISEL-X86-LABEL: pow_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $28, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fld %st(0)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    fstpt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    calll powl
; GISEL-X86-NEXT:    addl $28, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: pow_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $40, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fld %st(0)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    fstpt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    callq powl
; GISEL-X64-NEXT:    addq $40, %rsp
; GISEL-X64-NEXT:    retq
  %r = tail call nnan ninf x86_fp80 @llvm.pow.f80(x86_fp80 %x, x86_fp80 %x)
  ret x86_fp80 %r
}

declare float @llvm.exp.f32(float) #1
declare double @llvm.exp.f64(double) #1
declare x86_fp80 @llvm.exp.f80(x86_fp80) #1

declare float @llvm.exp2.f32(float) #1
declare double @llvm.exp2.f64(double) #1
declare x86_fp80 @llvm.exp2.f80(x86_fp80) #1

declare float @llvm.log.f32(float) #1
declare double @llvm.log.f64(double) #1
declare x86_fp80 @llvm.log.f80(x86_fp80) #1

declare float @llvm.log2.f32(float) #1
declare double @llvm.log2.f64(double) #1
declare x86_fp80 @llvm.log2.f80(x86_fp80) #1

declare float @llvm.log10.f32(float) #1
declare double @llvm.log10.f64(double) #1
declare x86_fp80 @llvm.log10.f80(x86_fp80) #1

declare float @llvm.pow.f32(float, float) #1
declare double @llvm.pow.f64(double, double) #1
declare x86_fp80 @llvm.pow.f80(x86_fp80, x86_fp80) #1

attributes #0 = { nounwind "no-infs-fp-math"="true" "no-nans-fp-math"="true" }
attributes #1 = { nounwind readnone speculatable }

