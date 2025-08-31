// RUN: mplang-opt %s | FileCheck %s

// CHECK-LABEL: func @ctrl
func.func @ctrl(%c: i1, %x: i32) -> i32 {
  // CHECK: mpir.if
  %0 = "mpir.if"(%c, %x) ({
    // then
    %t = arith.addi %x, %x : i32
    mpir.yield %t : i32
  }, {
    // else
    %e = arith.subi %x, %x : i32
    mpir.yield %e : i32
  }) : (i1, i32) -> (i32)

  // CHECK: mpir.while
  %1 = "mpir.while"(%0) ({
    // cond
    %one = arith.constant 1 : i32
    %lt = arith.cmpi slt, %0, %one : i32
    mpir.cond_yield %lt : i1
  }, {
    // body
    %two = arith.constant 2 : i32
    %n = arith.addi %0, %two : i32
    mpir.yield %n : i32
  }) : (i32) -> (i32)
  return %1 : i32
}
