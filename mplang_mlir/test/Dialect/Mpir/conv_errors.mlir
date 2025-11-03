// RUN: mpir-opt %s -split-input-file -verify-diagnostics

// Test ConvOp pmask disjointness checking

module {
  func.func @valid_conv_disjoint(%a: !mpir.mp<tensor<10xf32>, 1>,
                                 %b: !mpir.mp<tensor<10xf32>, 2>)
                                 -> !mpir.mp<tensor<10xf32>, 3> {
    // Valid: pmask 1 and 2 are disjoint, union is 3
    %result = mpir.conv(%a, %b)
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>)
              -> !mpir.mp<tensor<10xf32>, 3>
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }
}

// -----

module {
  func.func @invalid_conv_overlapping(%a: !mpir.mp<tensor<10xf32>, 3>,
                                      %b: !mpir.mp<tensor<10xf32>, 5>)
                                      -> !mpir.mp<tensor<10xf32>, 7> {
    // Invalid: pmask 3 (0b011) and 5 (0b101) overlap at bit 0
    // expected-error @+1 {{input pmasks must be disjoint, but pmask 3 and pmask 5 overlap (intersection = 1)}}
    %result = mpir.conv(%a, %b)
              : (!mpir.mp<tensor<10xf32>, 3>, !mpir.mp<tensor<10xf32>, 5>)
              -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @invalid_conv_wrong_union(%a: !mpir.mp<tensor<10xf32>, 1>,
                                      %b: !mpir.mp<tensor<10xf32>, 2>)
                                      -> !mpir.mp<tensor<10xf32>, 7> {
    // Invalid: union of 1 and 2 is 3, not 7
    // expected-error @+1 {{result pmask must equal union of input pmasks. Expected 3, got 7}}
    %result = mpir.conv(%a, %b)
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>)
              -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @invalid_conv_inner_type_mismatch(%a: !mpir.mp<tensor<10xf32>, 1>,
                                              %b: !mpir.mp<tensor<10xi32>, 2>)
                                              -> !mpir.mp<tensor<10xf32>, 3> {
    // Invalid: inner types don't match (f32 vs i32)
    // expected-error @+1 {{all inputs must have same inner type, got MP<tensor<10xf32>> and MP<tensor<10xi32>>}}
    %result = mpir.conv(%a, %b)
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xi32>, 2>)
              -> !mpir.mp<tensor<10xf32>, 3>
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }
}

// -----

module {
  func.func @valid_conv_three_way(%a: !mpir.mp<tensor<10xf32>, 1>,
                                  %b: !mpir.mp<tensor<10xf32>, 2>,
                                  %c: !mpir.mp<tensor<10xf32>, 4>)
                                  -> !mpir.mp<tensor<10xf32>, 7> {
    // Valid: pmasks 1, 2, 4 are pairwise disjoint, union is 7
    %result = mpir.conv(%a, %b, %c)
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>, !mpir.mp<tensor<10xf32>, 4>)
              -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}
