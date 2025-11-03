// RUN: mpir-opt %s -split-input-file -verify-diagnostics

// Test error detection in uniform_cond operation

module {
  func.func @non_scalar_condition(%pred: !mpir.mp<tensor<10xi1>, 7>,
                                  %a: !mpir.mp<tensor<10xf32>, 7>,
                                  %b: !mpir.mp<tensor<10xf32>, 7>)
                                  -> !mpir.mp<tensor<10xf32>, 7> {
    // expected-error @+1 {{condition must be MP<i1, pmask>, got MP<tensor<10xi1>, pmask>}}
    %result = mpir.uniform_cond %pred : !mpir.mp<tensor<10xi1>, 7>
              -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @non_boolean_condition(%pred: !mpir.mp<i32, 7>,
                                   %a: !mpir.mp<tensor<10xf32>, 7>,
                                   %b: !mpir.mp<tensor<10xf32>, 7>)
                                   -> !mpir.mp<tensor<10xf32>, 7> {
    // expected-error @+1 {{condition must have inner type i1 (boolean), got i32}}
    %result = mpir.uniform_cond %pred : !mpir.mp<i32, 7>
              -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @mismatched_branch_types(%pred: !mpir.mp<i1, 7>,
                                     %a: !mpir.mp<tensor<10xf32>, 7>,
                                     %b: !mpir.mp<tensor<20xf32>, 7>)
                                     -> !mpir.mp<tensor<10xf32>, 7> {
    // expected-error @+1 {{then branch returns type !mpir.mp<tensor<10xf32>, 7> at position 0, but else branch returns type !mpir.mp<tensor<20xf32>, 7>}}
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7>
              -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %b : !mpir.mp<tensor<20xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @mismatched_result_count(%pred: !mpir.mp<i1, 7>,
                                     %a: !mpir.mp<tensor<10xf32>, 7>)
                                     -> !mpir.mp<tensor<10xf32>, 7> {
    // expected-error @+1 {{then branch returns 1 values, but else branch returns 0 values}}
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7>
              -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}
