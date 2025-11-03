// SPMD OpSet test for Mplang dialect
// RUN: mplang-opt %s | FileCheck %s

// CHECK-LABEL: @test_peval_mlir_func
func.func @test_peval_mlir_func(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // Test peval with MLIR function reference (Mode 1) - parties 0,1,2 execute
  // CHECK: mplang.peval @compute
  // CHECK-SAME: mask = 7
  %0 = mplang.peval @compute(%arg0, %arg1) {mask = 7 : i64}
       : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_peval_external
func.func @test_peval_external(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // Test peval with external backend (Mode 2) - party 0 executes
  // CHECK: mplang.peval
  // CHECK-SAME: fn_type = "phe.encrypt"
  // CHECK-SAME: mask = 1
  %0 = mplang.peval (%arg0) {fn_type = "phe.encrypt", mask = 1 : i64}
       : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_peval_with_fn_attrs
func.func @test_peval_with_fn_attrs() -> (tensor<1xi8>, tensor<1xi8>) {
  // Test peval with fn_attrs for backend parameters
  // CHECK: mplang.peval
  // CHECK-SAME: fn_attrs = {key_size = 2048 : i64, scheme = "paillier"}
  // CHECK-SAME: fn_type = "phe.keygen"
  %pk, %sk = mplang.peval () {
    fn_type = "phe.keygen",
    fn_attrs = {scheme = "paillier", key_size = 2048 : i64},
    mask = 1 : i64
  } : () -> (tensor<1xi8>, tensor<1xi8>)
  func.return %pk, %sk : tensor<1xi8>, tensor<1xi8>
}

// CHECK-LABEL: @test_peval_dyn_mlir
func.func @test_peval_dyn_mlir(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // Test peval_dyn with MLIR function - execution determined by arg availability
  // CHECK: mplang.peval_dyn @compute
  %0 = mplang.peval_dyn @compute(%arg0, %arg1)
       : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @test_peval_dyn_external
func.func @test_peval_dyn_external(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // Test peval_dyn with external backend
  // CHECK: mplang.peval_dyn
  // CHECK-SAME: fn_type = "basic.identity"
  %0 = mplang.peval_dyn (%arg0) {fn_type = "basic.identity"}
       : (tensor<10xf32>) -> tensor<10xf32>
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
