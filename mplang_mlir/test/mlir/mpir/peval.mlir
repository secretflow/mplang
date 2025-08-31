// RUN: mplang-opt --pass-pipeline="builtin.module(mpir-peval)" %s | FileCheck %s

// CHECK-LABEL: func @peval_tuple_fold
func.func @peval_tuple_fold(%x: i32) -> i32 {
  // Single-element tuple then pconv should fold to identity
  // CHECK: return %{{.*}} : i32
  %0 = "mpir.tuple"(%x) : (i32) -> (i32)
  %1 = "mpir.pconv"(%0#0) : (i32) -> i32
  return %1 : i32
}
