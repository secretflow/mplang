// RUN: mpir-opt %s | FileCheck %s

// Example: Control Flow Patterns and Best Practices
// This file demonstrates correct usage of uniform_cond and uniform_while
// operations in the Mpir dialect, along with common patterns and pitfalls.

module {
  // ============================================================================
  // Pattern 1: Basic uniform_cond
  // ============================================================================

  // CORRECT: Simple conditional with scalar boolean condition
  func.func @pattern_basic_cond(%cond: !mpir.mp<i1, 7>,
                                %a: !mpir.mp<tensor<10xf32>, 7>,
                                %b: !mpir.mp<tensor<10xf32>, 7>)
                                -> !mpir.mp<tensor<10xf32>, 7> {
    %result = mpir.uniform_cond %cond : !mpir.mp<i1, 7>
              -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // INCORRECT: Condition must be scalar, not tensor
  // This will fail verification:
  //   func.func @wrong_tensor_condition(%cond: !mpir.mp<tensor<10xi1>, 7>, ...)
  //   %result = mpir.uniform_cond %cond : !mpir.mp<tensor<10xi1>, 7>
  // Error: condition must be MP<i1, pmask>, got MP<tensor<10xi1>, pmask>

  // ============================================================================
  // Pattern 2: Nested conditional with complex branches
  // ============================================================================

  func.func @pattern_nested_cond(%outer_cond: !mpir.mp<i1, 3>,
                                 %inner_cond: !mpir.mp<i1, 3>,
                                 %x: !mpir.mp<tensor<10xf32>, 3>,
                                 %y: !mpir.mp<tensor<10xf32>, 3>,
                                 %z: !mpir.mp<tensor<10xf32>, 3>)
                                 -> !mpir.mp<tensor<10xf32>, 3> {
    %result = mpir.uniform_cond %outer_cond : !mpir.mp<i1, 3>
              -> !mpir.mp<tensor<10xf32>, 3> {
      // Then branch: another conditional
      %inner_result = mpir.uniform_cond %inner_cond : !mpir.mp<i1, 3>
                      -> !mpir.mp<tensor<10xf32>, 3> {
        mpir.return %x : !mpir.mp<tensor<10xf32>, 3>
      } {
        mpir.return %y : !mpir.mp<tensor<10xf32>, 3>
      }
      mpir.return %inner_result : !mpir.mp<tensor<10xf32>, 3>
    } {
      // Else branch: direct return
      mpir.return %z : !mpir.mp<tensor<10xf32>, 3>
    }
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }

  // ============================================================================
  // Pattern 3: Conditional with computation in branches
  // ============================================================================

  func.func @pattern_cond_with_computation(%pred: !mpir.mp<i1, 7>,
                                           %data: !mpir.mp<tensor<100xf32>, 7>)
                                           -> !mpir.mp<tensor<100xf32>, 7> {
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7>
              -> !mpir.mp<tensor<100xf32>, 7> {
      // Expensive computation only executed if condition is true
      %processed = mpir.peval @expensive_transform(%data) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<100xf32>, 7>)
        -> !mpir.mp<tensor<100xf32>, 7>
      mpir.return %processed : !mpir.mp<tensor<100xf32>, 7>
    } {
      // Cheap computation for false case
      %processed = mpir.peval @simple_transform(%data) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<100xf32>, 7>)
        -> !mpir.mp<tensor<100xf32>, 7>
      mpir.return %processed : !mpir.mp<tensor<100xf32>, 7>
    }
    return %result : !mpir.mp<tensor<100xf32>, 7>
  }

  // ============================================================================
  // Pattern 4: Basic while loop
  // ============================================================================

  func.func @pattern_basic_while(%init: !mpir.mp<tensor<10xf32>, 7>,
                                 %max_iters: !mpir.mp<i32, 7>)
                                 -> !mpir.mp<tensor<10xf32>, 7> {
    // Initialize counter
    %c0 = mpir.peval () {
      fn_type = "const",
      fn_attrs = {value = 0 : i32},
      rmask = 7 : i64
    } : () -> !mpir.mp<i32, 7>

    // Loop with iteration counter
    %final_data, %final_iter = mpir.uniform_while (%init, %c0)
                               : (!mpir.mp<tensor<10xf32>, 7>, !mpir.mp<i32, 7>)
                               -> (!mpir.mp<tensor<10xf32>, 7>, !mpir.mp<i32, 7>) {
    ^bb0(%data: !mpir.mp<tensor<10xf32>, 7>, %iter: !mpir.mp<i32, 7>):
      // Condition: continue while iter < max_iters
      %cond = mpir.peval @less_than(%iter, %max_iters) {
        rmask = 7 : i64
      } : (!mpir.mp<i32, 7>, !mpir.mp<i32, 7>)
        -> !mpir.mp<i1, 7>

      mpir.condition %cond : !mpir.mp<i1, 7>
    } do {
    ^bb0(%data: !mpir.mp<tensor<10xf32>, 7>, %iter: !mpir.mp<i32, 7>):
      // Loop body: process data
      %new_data = mpir.peval @process_step(%data) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<10xf32>, 7>)
        -> !mpir.mp<tensor<10xf32>, 7>

      // Increment counter
      %c1 = mpir.peval () {
        fn_type = "const",
        fn_attrs = {value = 1 : i32},
        rmask = 7 : i64
      } : () -> !mpir.mp<i32, 7>

      %new_iter = mpir.peval @add_i32(%iter, %c1) {
        rmask = 7 : i64
      } : (!mpir.mp<i32, 7>, !mpir.mp<i32, 7>)
        -> !mpir.mp<i32, 7>

      mpir.yield %new_data, %new_iter
              : !mpir.mp<tensor<10xf32>, 7>, !mpir.mp<i32, 7>
    }

    return %final_data : !mpir.mp<tensor<10xf32>, 7>
  }

  // ============================================================================
  // Pattern 5: Convergence-based while loop
  // ============================================================================

  func.func @pattern_convergence_while(%init: !mpir.mp<tensor<10xf32>, 7>,
                                       %tolerance: !mpir.mp<f32, 7>)
                                       -> !mpir.mp<tensor<10xf32>, 7> {
    // Loop until convergence
    %result = mpir.uniform_while (%init) : (!mpir.mp<tensor<10xf32>, 7>)
                                         -> (!mpir.mp<tensor<10xf32>, 7>) {
    ^bb0(%x: !mpir.mp<tensor<10xf32>, 7>):
      // Compute next iteration
      %x_next = mpir.peval @compute_iteration(%x) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<10xf32>, 7>)
        -> !mpir.mp<tensor<10xf32>, 7>

      // Check convergence: ||x_next - x|| > tolerance
      %delta = mpir.peval @compute_delta(%x, %x_next) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<10xf32>, 7>, !mpir.mp<tensor<10xf32>, 7>)
        -> !mpir.mp<f32, 7>

      %not_converged = mpir.peval @greater_than(%delta, %tolerance) {
        rmask = 7 : i64
      } : (!mpir.mp<f32, 7>, !mpir.mp<f32, 7>)
        -> !mpir.mp<i1, 7>

      mpir.condition %not_converged : !mpir.mp<i1, 7>
    } do {
    ^bb0(%x: !mpir.mp<tensor<10xf32>, 7>):
      // Compute next value
      %x_next = mpir.peval @compute_iteration(%x) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<10xf32>, 7>)
        -> !mpir.mp<tensor<10xf32>, 7>

      mpir.yield %x_next : !mpir.mp<tensor<10xf32>, 7>
    }

    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // ============================================================================
  // Pattern 6: verify_uniform usage
  // ============================================================================

  // RECOMMENDED: Enable runtime verification for MPC computations
  func.func @pattern_verify_uniform_enabled(%cond: !mpir.mp<i1, 3>,
                                            %a: !mpir.mp<tensor<10xf32>, 3>,
                                            %b: !mpir.mp<tensor<10xf32>, 3>)
                                            -> !mpir.mp<tensor<10xf32>, 3> {
    // verify_uniform=true: Runtime checks condition is same across parties
    %result = mpir.uniform_cond %cond {verify_uniform = true}
              : !mpir.mp<i1, 3> -> !mpir.mp<tensor<10xf32>, 3> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 3>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 3>
    }
    return %result : !mpir.mp<tensor<10xf32>, 3>
  }

  // Use verify_uniform=false only when:
  // 1. Performance is critical and condition is guaranteed uniform
  // 2. Condition comes from trusted source (e.g., controller)
  // 3. Already verified at higher level
  func.func @pattern_verify_uniform_disabled(%cond: !mpir.mp<i1, 7>,
                                             %a: !mpir.mp<tensor<10xf32>, 7>,
                                             %b: !mpir.mp<tensor<10xf32>, 7>)
                                             -> !mpir.mp<tensor<10xf32>, 7> {
    // Trusted condition from controller, skip runtime verification
    %result = mpir.uniform_cond %cond {verify_uniform = false}
              : !mpir.mp<i1, 7> -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // ============================================================================
  // Pattern 7: Type consistency requirements
  // ============================================================================

  // CORRECT: Both branches return same type
  func.func @pattern_type_correct(%pred: !mpir.mp<i1, 7>,
                                  %x: !mpir.mp<tensor<10xf32>, 7>,
                                  %y: !mpir.mp<tensor<10xf32>, 7>)
                                  -> !mpir.mp<tensor<10xf32>, 7> {
    %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7>
              -> !mpir.mp<tensor<10xf32>, 7> {
      mpir.return %x : !mpir.mp<tensor<10xf32>, 7>
    } {
      mpir.return %y : !mpir.mp<tensor<10xf32>, 7>
    }
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // INCORRECT: Branch type mismatch
  // This will fail verification:
  //   %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7>
  //           -> !mpir.mp<tensor<10xf32>, 7> {
  //     mpir.return %x : !mpir.mp<tensor<10xf32>, 7>
  //   } {
  //     mpir.return %z : !mpir.mp<tensor<20xf32>, 7>  // Wrong shape!
  //   }
  // Error: then branch returns type '!mpir.mp<tensor<10xf32>, 7>' at position 0,
  //        but else branch returns type '!mpir.mp<tensor<20xf32>, 7>'
}

// ============================================================================
// Best Practices Summary
// ============================================================================
//
// 1. Condition Requirements:
//    ✓ Must be scalar MP<i1, pmask> or MPDynamic<i1>
//    ✗ Cannot use tensor<Nxi1> (element-wise conditions not supported)
//    ✓ All parties must have same boolean value (enforced by MPC or verify_uniform)
//
// 2. Branch Type Matching:
//    ✓ Both branches must return exactly same types
//    ✓ Number of return values must match
//    ✓ Types are compared structurally (including pmask)
//
// 3. While Loop Requirements:
//    ✓ Condition region must return MP<i1, pmask> or MPDynamic<i1>
//    ✓ Body region must yield same types as init arguments
//    ✓ Result types must match init types
//    ✓ Both regions must have proper terminators (condition/yield)
//
// 4. verify_uniform Attribute:
//    ✓ Default: true (safe, recommended for MPC)
//    ✓ Set false only when condition is guaranteed uniform
//    ✓ Runtime verification uses all-gather communication
//
// 5. Performance Tips:
//    ✓ Minimize computation in condition regions (executed every iteration)
//    ✓ Hoist invariant computations outside loops
//    ✓ Use verify_uniform=false carefully for performance-critical code
//    ✓ Consider early exit patterns for convergence loops
//
// 6. Security Considerations:
//    ✓ Uniform control flow prevents timing attacks
//    ✓ Both branches should have similar execution patterns
//    ✓ Enable verify_uniform for untrusted condition sources
//    ✓ Condition computation should be MPC-secure (use peval with proper rmask)
