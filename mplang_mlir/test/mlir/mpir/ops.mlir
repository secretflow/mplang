// RUN: mplang-opt %s | FileCheck %s

// CHECK-LABEL: func @basic
func.func @basic(%arg0: i32, %arg1: i32) {
  // CHECK: mpir.tuple
  %0:2 = "mpir.tuple"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  // CHECK: mpir.pconv
  %1 = "mpir.pconv"(%0#0, %0#1) : (i32, i32) -> i32
  // CHECK: mpir.shfl_static
  %2 = "mpir.shfl_static"(%1) {pmask = 3 : i64, src_ranks = [0, 1]} : (i32) -> i32
  return
}
