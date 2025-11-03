// SPMD OpSet test for MPLANG dialect
// RUN: mplang-opt %s | FileCheck %s

// CHECK-LABEL: @test_pcall_static
func.func @test_pcall_static(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // Test static mask private call - parties 0,1,2 execute
  // CHECK: mplang.pcall @compute
  // CHECK-SAME: mask = 7
  %0 = mplang.pcall @compute(%arg0, %arg1) {mask = 7 : i64}
       : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_pcall_dyn
func.func @test_pcall_dyn(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // Test dynamic private call - execution determined by arg availability
  // CHECK: mplang.pcall_dyn @compute
  %0 = mplang.pcall_dyn @compute(%arg0, %arg1)
       : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_shfl_static
func.func @test_shfl_static(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // Test static shuffle - parties 0,2 receive from ranks 1,3
  // CHECK: mplang.shfl
  // CHECK-SAME: mask = 5
  // CHECK-SAME: src_ranks = array<i64: 1, 3>
  %0 = mplang.shfl %arg0 {mask = 5 : i64, src_ranks = array<i64: 1, 3>}
       : tensor<10xf32> -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_shfl_dyn
func.func @test_shfl_dyn(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // Test dynamic shuffle
  // CHECK: mplang.shfl_dyn
  %mask = arith.constant 5 : i64
  %c1 = arith.constant 1 : i64
  %c3 = arith.constant 3 : i64
  %ranks = tensor.from_elements %c1, %c3 : tensor<2xi64>
  %0 = mplang.shfl_dyn %arg0, %mask, %ranks
       : tensor<10xf32>, i64, tensor<2xi64> -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_conv
func.func @test_conv(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // Test convergence - merge disjoint values
  // Party 0 has arg0 (pmask=0b001), party 2 has arg1 (pmask=0b100)
  // Result has pmask=0b101, each party retains its data
  // CHECK: mplang.conv
  %0 = mplang.conv(%arg0, %arg1)
       : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// Helper function for call tests
// CHECK-LABEL: @compute
func.func @compute(%a: tensor<10xf32>, %b: tensor<10xf32>) -> tensor<10xf32> {
  func.return %a : tensor<10xf32>
}
