// RUN: mpir-opt %s -split-input-file -verify-diagnostics

// Test error detection in peval operation (rmask validation)

module {
  func.func @valid_rmask_subset(%x: !mpir.mp<tensor<10xf32>, 7>)
                                -> !mpir.mp<tensor<10xf32>, 7> {
    // Valid: rmask=7 is subset of input pmask=7
    %result = mpir.peval @compute(%x) {rmask = 7 : i64}
              : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @valid_rmask_subset_narrowing(%x: !mpir.mp<tensor<10xf32>, 7>)
                                          -> !mpir.mp<tensor<10xf32>, 3> {
    // Valid: rmask=3 (parties 0,1) is subset of input pmask=7 (parties 0,1,2)
    %result = mpir.peval @compute(%x) {rmask = 3 : i64}
              : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 3>
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }
}

// -----

module {
  func.func @invalid_rmask_not_subset(%x: !mpir.mp<tensor<10xf32>, 3>)
                                      -> !mpir.mp<tensor<10xf32>, 7> {
    // Invalid: rmask=7 (parties 0,1,2) is NOT subset of input pmask=3 (parties 0,1)
    // Party 2 doesn't have input data but rmask says it should execute
    // expected-error @+1 {{rmask 7 is not a subset of input pmask union 3}}
    %result = mpir.peval @compute(%x) {rmask = 7 : i64}
              : (!mpir.mp<tensor<10xf32>, 3>) -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @invalid_rmask_disjoint(%x: !mpir.mp<tensor<10xf32>, 3>)
                                    -> !mpir.mp<tensor<10xf32>, 4> {
    // Invalid: rmask=4 (party 2) is completely disjoint from input pmask=3 (parties 0,1)
    // Party 2 doesn't have input data
    // expected-error @+1 {{rmask 4 is not a subset of input pmask union 3}}
    %result = mpir.peval @compute(%x) {rmask = 4 : i64}
              : (!mpir.mp<tensor<10xf32>, 3>) -> !mpir.mp<tensor<10xf32>, 4>
    return %result : !mpir.mp<tensor<10xf32>, 4>
  }
}

// -----

module {
  func.func @valid_rmask_union_of_inputs(%x: !mpir.mp<tensor<10xf32>, 1>,
                                         %y: !mpir.mp<tensor<10xf32>, 2>)
                                         -> !mpir.mp<tensor<10xf32>, 3> {
    // Valid: rmask=3 equals union of input pmasks (1 | 2 = 3)
    %result = mpir.peval @add(%x, %y) {rmask = 3 : i64}
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>)
              -> !mpir.mp<tensor<10xf32>, 3>
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }
}

// -----

module {
  func.func @invalid_rmask_exceeds_union(%x: !mpir.mp<tensor<10xf32>, 1>,
                                         %y: !mpir.mp<tensor<10xf32>, 2>)
                                         -> !mpir.mp<tensor<10xf32>, 7> {
    // Invalid: rmask=7 (parties 0,1,2) exceeds union of inputs (1 | 2 = 3, parties 0,1)
    // Party 2 doesn't have any input data
    // expected-error @+1 {{rmask 7 is not a subset of input pmask union 3}}
    %result = mpir.peval @add(%x, %y) {rmask = 7 : i64}
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>)
              -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @valid_no_rmask(%x: !mpir.mp<tensor<10xf32>, 7>)
                            -> !mpir.mp<tensor<10xf32>, 7> {
    // Valid: No rmask specified, defaults to input pmask
    %result = mpir.peval @compute(%x)
              : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @valid_keygen_no_input() -> !mpir.mp<tensor<0xi8>, 1> {
    // Valid: rmask=1 with no inputs (keygen scenario)
    %pk = mpir.peval () {
      fn_type = "phe.keygen",
      fn_attrs = {scheme = "paillier"},
      rmask = 1 : i64
    } : () -> !mpir.mp<tensor<0xi8>, 1>
    return %pk : !mpir.mp<tensor<0xi8>, 1>
  }
}
