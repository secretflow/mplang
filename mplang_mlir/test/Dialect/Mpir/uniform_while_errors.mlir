// RUN: mpir-opt %s -split-input-file -verify-diagnostics

// Test error detection in uniform_while operation

module {
  func.func @non_scalar_condition(%init: !mpir.mp<tensor<i32>, 7>,
                                  %cond_tensor: !mpir.mp<tensor<10xi1>, 7>)
                                  -> !mpir.mp<tensor<i32>, 7> {
    // expected-error @+1 {{condition must be MP<i1, pmask>, got MP<'tensor<10xi1>', pmask>}}
    %result = mpir.uniform_while (%init) : (!mpir.mp<tensor<i32>, 7>)
              -> !mpir.mp<tensor<i32>, 7> {
      ^cond(%state: !mpir.mp<tensor<i32>, 7>):
        mpir.condition %cond_tensor : !mpir.mp<tensor<10xi1>, 7>
    } do {
      ^body(%state: !mpir.mp<tensor<i32>, 7>):
        mpir.yield %state : !mpir.mp<tensor<i32>, 7>
    }
    return %result : !mpir.mp<tensor<i32>, 7>
  }
}

// -----

module {
  func.func @non_boolean_condition(%init: !mpir.mp<tensor<i32>, 7>,
                                   %cond_int: !mpir.mp<i32, 7>)
                                   -> !mpir.mp<tensor<i32>, 7> {
    // expected-error @+1 {{condition must have inner type i1 (boolean), got i32}}
    %result = mpir.uniform_while (%init) : (!mpir.mp<tensor<i32>, 7>)
              -> !mpir.mp<tensor<i32>, 7> {
      ^cond(%state: !mpir.mp<tensor<i32>, 7>):
        mpir.condition %cond_int : !mpir.mp<i32, 7>
    } do {
      ^body(%state: !mpir.mp<tensor<i32>, 7>):
        mpir.yield %state : !mpir.mp<tensor<i32>, 7>
    }
    return %result : !mpir.mp<tensor<i32>, 7>
  }
}

// -----

module {
  func.func @yield_type_mismatch(%init: !mpir.mp<tensor<i32>, 7>)
                                 -> !mpir.mp<tensor<i32>, 7> {
    // expected-error @+1 {{init arg at position 0 has type '!mpir.mp<tensor<i32>, 7>', but body yields type '!mpir.mp<tensor<i64>, 7>'}}
    %result = mpir.uniform_while (%init) : (!mpir.mp<tensor<i32>, 7>)
              -> !mpir.mp<tensor<i32>, 7> {
      ^cond(%state: !mpir.mp<tensor<i32>, 7>):
        %cond = mpir.peval @check(%state) {rmask = 7 : i64}
                : (!mpir.mp<tensor<i32>, 7>) -> !mpir.mp<i1, 7>
        mpir.condition %cond : !mpir.mp<i1, 7>
    } do {
      ^body(%state: !mpir.mp<tensor<i32>, 7>):
        %wrong_type = mpir.peval @convert_i64(%state) {rmask = 7 : i64}
                      : (!mpir.mp<tensor<i32>, 7>) -> !mpir.mp<tensor<i64>, 7>
        mpir.yield %wrong_type : !mpir.mp<tensor<i64>, 7>
    }
    return %result : !mpir.mp<tensor<i32>, 7>
  }
}

// -----

module {
  func.func @result_type_mismatch(%init: !mpir.mp<tensor<i32>, 7>)
                                  -> !mpir.mp<tensor<f32>, 7> {
    // expected-error @+1 {{init arg at position 0 has type '!mpir.mp<tensor<i32>, 7>', but result has type '!mpir.mp<tensor<f32>, 7>'}}
    %result = mpir.uniform_while (%init) : (!mpir.mp<tensor<i32>, 7>)
              -> !mpir.mp<tensor<f32>, 7> {
      ^cond(%state: !mpir.mp<tensor<i32>, 7>):
        %cond = mpir.peval @check(%state) {rmask = 7 : i64}
                : (!mpir.mp<tensor<i32>, 7>) -> !mpir.mp<i1, 7>
        mpir.condition %cond : !mpir.mp<i1, 7>
    } do {
      ^body(%state: !mpir.mp<tensor<i32>, 7>):
        mpir.yield %state : !mpir.mp<tensor<i32>, 7>
    }
    return %result : !mpir.mp<tensor<f32>, 7>
  }
}

// -----

module {
  func.func @yield_count_mismatch(%init1: !mpir.mp<tensor<i32>, 7>,
                                  %init2: !mpir.mp<tensor<f32>, 7>)
                                  -> (!mpir.mp<tensor<i32>, 7>, !mpir.mp<tensor<f32>, 7>) {
    // expected-error @+1 {{loop initialized with 2 values, but body yields 1 values}}
    %r1, %r2 = mpir.uniform_while (%init1, %init2)
               : (!mpir.mp<tensor<i32>, 7>, !mpir.mp<tensor<f32>, 7>)
               -> (!mpir.mp<tensor<i32>, 7>, !mpir.mp<tensor<f32>, 7>) {
      ^cond(%s1: !mpir.mp<tensor<i32>, 7>, %s2: !mpir.mp<tensor<f32>, 7>):
        %cond = mpir.peval @check(%s1) {rmask = 7 : i64}
                : (!mpir.mp<tensor<i32>, 7>) -> !mpir.mp<i1, 7>
        mpir.condition %cond : !mpir.mp<i1, 7>
    } do {
      ^body(%s1: !mpir.mp<tensor<i32>, 7>, %s2: !mpir.mp<tensor<f32>, 7>):
        // Only yield one value instead of two
        mpir.yield %s1 : !mpir.mp<tensor<i32>, 7>
    }
    return %r1, %r2 : !mpir.mp<tensor<i32>, 7>, !mpir.mp<tensor<f32>, 7>
  }
}
