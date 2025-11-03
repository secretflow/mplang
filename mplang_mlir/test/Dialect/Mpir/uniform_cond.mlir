// RUN: mpir-opt %s | mpir-opt | FileCheck %s

// Test uniform_cond operation

module {
  // CHECK-LABEL: func.func @simple_uniform_cond
  func.func @simple_uniform_cond(%pred: !mpir.mp<i1, 7>,
                                 %a: !mpir.mp<tensor<10xf32>, 7>,
                                 %b: !mpir.mp<tensor<10xf32>, 7>)
                                 -> !mpir.mp<tensor<10xf32>, 7> {
    // CHECK: mpir.uniform_cond %{{.*}} : !mpir.mp<i1, 7> -> !mpir.mp<tensor<10xf32>, 7>
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7> -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // CHECK-LABEL: func.func @uniform_cond_with_verify_disabled
  func.func @uniform_cond_with_verify_disabled(%pred: !mpir.mp<i1, 3>,
                                                %a: !mpir.mp<tensor<10xf32>, 3>,
                                                %b: !mpir.mp<tensor<10xf32>, 3>)
                                                -> !mpir.mp<tensor<10xf32>, 3> {
    // CHECK: mpir.uniform_cond %{{.*}} {verify_uniform = false}
    %result = mpir.uniform_cond %pred {verify_uniform = false}
              : !mpir.mp<i1, 3> -> !mpir.mp<tensor<10xf32>, 3> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 3>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 3>
    }
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }

  // CHECK-LABEL: func.func @uniform_cond_encrypted
  func.func @uniform_cond_encrypted(%pred: !mpir.mp<i1, 3>,
                                    %enc_a: !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>,
                                    %enc_b: !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>)
                                    -> !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3> {
    // CHECK: mpir.uniform_cond %{{.*}} : !mpir.mp<i1, 3> -> !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 3>
              -> !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3> {
      mpir.return %enc_a : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>
    } {
      mpir.return %enc_b : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>
    }
    return %result : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>
  }

  // CHECK-LABEL: func.func @uniform_cond_with_peval
  func.func @uniform_cond_with_peval(%pred: !mpir.mp<i1, 7>,
                                     %x: !mpir.mp<tensor<10xf32>, 7>,
                                     %y: !mpir.mp<tensor<10xf32>, 7>)
                                     -> !mpir.mp<tensor<10xf32>, 7> {
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7> -> !mpir.mp<tensor<10xf32>, 7> {
      // Then branch: call expensive function
      %a = mpir.peval @expensive_compute(%x) {rmask = 7 : i64}
           : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7>
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      // Else branch: call cheap function
      %b = mpir.peval @cheap_compute(%y) {rmask = 7 : i64}
           : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7>
      mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}
