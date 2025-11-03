// RUN: mpir-opt %s | mpir-opt | FileCheck %s

// Test type parsing and printing for Mpir types

module {
  // CHECK-LABEL: func.func @test_mp_type
  func.func @test_mp_type(%arg0: !mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7> {
    // CHECK: %arg0 : !mpir.mp<tensor<10xf32>, 7>
    return %arg0 : !mpir.mp<tensor<10xf32>, 7>
  }

  // CHECK-LABEL: func.func @test_encrypted_type
  func.func @test_encrypted_type(%arg0: !mpir.enc<tensor<10xf32>, "paillier">)
                                 -> !mpir.enc<tensor<10xf32>, "paillier"> {
    // CHECK: %arg0 : !mpir.enc<tensor<10xf32>, "paillier">
    return %arg0 : !mpir.enc<tensor<10xf32>, "paillier">
  }

  // CHECK-LABEL: func.func @test_mp_encrypted_type
  func.func @test_mp_encrypted_type(%arg0: !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>)
                                    -> !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3> {
    // CHECK: %arg0 : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>
    return %arg0 : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>
  }

  // CHECK-LABEL: func.func @test_mp_dynamic_type
  func.func @test_mp_dynamic_type(%arg0: !mpir.mp_dynamic<tensor<10xf32>>)
                                  -> !mpir.mp_dynamic<tensor<10xf32>> {
    // CHECK: %arg0 : !mpir.mp_dynamic<tensor<10xf32>>
    return %arg0 : !mpir.mp_dynamic<tensor<10xf32>>
  }

  // CHECK-LABEL: func.func @test_multiple_pmasks
  func.func @test_multiple_pmasks(%arg0: !mpir.mp<tensor<10xf32>, 1>,
                                  %arg1: !mpir.mp<tensor<10xf32>, 2>,
                                  %arg2: !mpir.mp<tensor<10xf32>, 4>) {
    // CHECK: %arg0 : !mpir.mp<tensor<10xf32>, 1>
    // CHECK: %arg1 : !mpir.mp<tensor<10xf32>, 2>
    // CHECK: %arg2 : !mpir.mp<tensor<10xf32>, 4>
    return
  }

  // CHECK-LABEL: func.func @test_mp_boolean
  func.func @test_mp_boolean(%arg0: !mpir.mp<i1, 7>) -> !mpir.mp<i1, 7> {
    // CHECK: %arg0 : !mpir.mp<i1, 7>
    return %arg0 : !mpir.mp<i1, 7>
  }

  // CHECK-LABEL: func.func @test_encrypted_ckks
  func.func @test_encrypted_ckks(%arg0: !mpir.enc<tensor<128xf32>, "ckks">)
                                 -> !mpir.enc<tensor<128xf32>, "ckks"> {
    // CHECK: %arg0 : !mpir.enc<tensor<128xf32>, "ckks">
    return %arg0 : !mpir.enc<tensor<128xf32>, "ckks">
  }
}
