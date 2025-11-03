// RUN: mpir-opt %s | FileCheck %s

// Example: Secure Multi-Party Computation (MPC) Pattern
// This example demonstrates secure computation on private data from multiple parties:
//   1. Each party holds private input data (not visible to others)
//   2. Data is converted to secret shares via conv operation
//   3. Computation happens on secret-shared data
//   4. Result can be revealed to one or all parties via conv

module {
  // Example 1: Secure sum - compute sum of private values
  // Party 0 has x0, Party 1 has x1, Party 2 has x2
  // Goal: Compute x0 + x1 + x2 without revealing individual values
  func.func @secure_sum(%x0: !mpir.mp<tensor<10xf32>, 1>,
                        %x1: !mpir.mp<tensor<10xf32>, 2>,
                        %x2: !mpir.mp<tensor<10xf32>, 4>)
                        -> !mpir.mp<tensor<10xf32>, 7> {
    // Convert private inputs to secret shares
    // conv: creates secret shares visible to all parties
    %shares = mpir.conv(%x0, %x1, %x2)
              : (!mpir.mp<tensor<10xf32>, 1>,
                 !mpir.mp<tensor<10xf32>, 2>,
                 !mpir.mp<tensor<10xf32>, 4>)
              -> !mpir.mp<tensor<10xf32>, 7>

    // Compute sum on secret shares
    // All parties participate (rmask=7 means parties 0,1,2)
    %result = mpir.peval @mpc_sum(%shares) {
      rmask = 7 : i64
    } : (!mpir.mp<tensor<10xf32>, 7>)
      -> !mpir.mp<tensor<10xf32>, 7>

    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // Example 2: Secure comparison with conditional output
  // Compare private values and return different results based on condition
  func.func @secure_compare(%x: !mpir.mp<tensor<10xf32>, 1>,
                            %y: !mpir.mp<tensor<10xf32>, 2>,
                            %threshold: !mpir.mp<f32, 7>)
                            -> !mpir.mp<tensor<10xf32>, 7> {
    // Create secret shares of inputs
    %x_shared = mpir.conv(%x) : (!mpir.mp<tensor<10xf32>, 1>)
                               -> !mpir.mp<tensor<10xf32>, 3>
    %y_shared = mpir.conv(%y) : (!mpir.mp<tensor<10xf32>, 2>)
                               -> !mpir.mp<tensor<10xf32>, 3>

    // Combine shares for joint computation
    %xy_shares = mpir.conv(%x_shared, %y_shared)
                 : (!mpir.mp<tensor<10xf32>, 3>,
                    !mpir.mp<tensor<10xf32>, 3>)
                 -> !mpir.mp<tensor<10xf32>, 3>

    // Compute difference securely
    %diff = mpir.peval @mpc_subtract(%xy_shares) {
      rmask = 3 : i64
    } : (!mpir.mp<tensor<10xf32>, 3>)
      -> !mpir.mp<tensor<10xf32>, 3>

    // Compare with threshold (produces boolean condition)
    %cond = mpir.peval @mpc_greater_than(%diff, %threshold) {
      rmask = 3 : i64
    } : (!mpir.mp<tensor<10xf32>, 3>, !mpir.mp<f32, 7>)
      -> !mpir.mp<i1, 3>

    // Conditional execution based on secure comparison
    // NOTE: uniform_cond requires all parties to have same boolean value
    // MPC protocols ensure condition is consistent across parties
    %result = mpir.uniform_cond %cond {verify_uniform = true}
              : !mpir.mp<i1, 3> -> !mpir.mp<tensor<10xf32>, 3> {
      // If x > y + threshold: return x
      mpir.return %x_shared : !mpir.mp<tensor<10xf32>, 3>
    } {
      // Otherwise: return y
      mpir.return %y_shared : !mpir.mp<tensor<10xf32>, 3>
    }

    // Share result with all parties
    %final_result = mpir.conv(%result)
                    : (!mpir.mp<tensor<10xf32>, 3>)
                    -> !mpir.mp<tensor<10xf32>, 7>

    return %final_result : !mpir.mp<tensor<10xf32>, 7>
  }

  // Example 3: Secure statistics - compute mean without revealing individual values
  func.func @secure_mean(%data0: !mpir.mp<tensor<100xf32>, 1>,
                         %data1: !mpir.mp<tensor<100xf32>, 2>)
                         -> !mpir.mp<f32, 3> {
    // Convert to secret shares
    %shares = mpir.conv(%data0, %data1)
              : (!mpir.mp<tensor<100xf32>, 1>,
                 !mpir.mp<tensor<100xf32>, 2>)
              -> !mpir.mp<tensor<100xf32>, 3>

    // Compute sum of all values
    %sum = mpir.peval @mpc_reduce_sum(%shares) {
      rmask = 3 : i64
    } : (!mpir.mp<tensor<100xf32>, 3>)
      -> !mpir.mp<f32, 3>

    // Compute count (200 total values)
    %count = mpir.peval () {
      fn_type = "const",
      fn_attrs = {value = 200.0 : f32},
      rmask = 3 : i64
    } : () -> !mpir.mp<f32, 3>

    // Divide sum by count to get mean
    %mean = mpir.peval @mpc_divide(%sum, %count) {
      rmask = 3 : i64
    } : (!mpir.mp<f32, 3>, !mpir.mp<f32, 3>)
      -> !mpir.mp<f32, 3>

    return %mean : !mpir.mp<f32, 3>
  }

  // Example 4: Iterative secure computation with while loop
  // Compute secure gradient descent iteration
  func.func @secure_gradient_descent(%x_init: !mpir.mp<tensor<10xf32>, 7>,
                                     %learning_rate: !mpir.mp<f32, 7>,
                                     %max_iters: !mpir.mp<i32, 7>)
                                     -> !mpir.mp<tensor<10xf32>, 7> {
    // Initialize iteration counter
    %c0 = mpir.peval () {
      fn_type = "const",
      fn_attrs = {value = 0 : i32},
      rmask = 7 : i64
    } : () -> !mpir.mp<i32, 7>

    // Iterative optimization loop
    %final_x, %final_iter = mpir.uniform_while (%x_init, %c0)
                            : (!mpir.mp<tensor<10xf32>, 7>, !mpir.mp<i32, 7>)
                            -> (!mpir.mp<tensor<10xf32>, 7>, !mpir.mp<i32, 7>) {
    ^bb0(%x: !mpir.mp<tensor<10xf32>, 7>, %iter: !mpir.mp<i32, 7>):
      // Condition: iter < max_iters
      %cond = mpir.peval @mpc_less_than(%iter, %max_iters) {
        rmask = 7 : i64
      } : (!mpir.mp<i32, 7>, !mpir.mp<i32, 7>)
        -> !mpir.mp<i1, 7>

      mpir.condition %cond : !mpir.mp<i1, 7>
    } do {
    ^bb0(%x: !mpir.mp<tensor<10xf32>, 7>, %iter: !mpir.mp<i32, 7>):
      // Compute gradient securely
      %gradient = mpir.peval @mpc_compute_gradient(%x) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<10xf32>, 7>)
        -> !mpir.mp<tensor<10xf32>, 7>

      // Update: x_new = x - learning_rate * gradient
      %update = mpir.peval @mpc_gradient_update(%x, %gradient, %learning_rate) {
        rmask = 7 : i64
      } : (!mpir.mp<tensor<10xf32>, 7>,
           !mpir.mp<tensor<10xf32>, 7>,
           !mpir.mp<f32, 7>)
        -> !mpir.mp<tensor<10xf32>, 7>

      // Increment iteration counter
      %c1 = mpir.peval () {
        fn_type = "const",
        fn_attrs = {value = 1 : i32},
        rmask = 7 : i64
      } : () -> !mpir.mp<i32, 7>

      %iter_next = mpir.peval @mpc_add_i32(%iter, %c1) {
        rmask = 7 : i64
      } : (!mpir.mp<i32, 7>, !mpir.mp<i32, 7>)
        -> !mpir.mp<i32, 7>

      mpir.yield %update, %iter_next
              : !mpir.mp<tensor<10xf32>, 7>, !mpir.mp<i32, 7>
    }

    return %final_x : !mpir.mp<tensor<10xf32>, 7>
  }
}

// Key MPC concepts demonstrated:
//
// 1. Secret Sharing via conv:
//    - Takes private inputs from different parties
//    - Produces secret shares visible to specified parties
//    - Ensures no single party can reconstruct original values
//
// 2. Party Participation (pmask semantics):
//    - pmask=1 (001): Only Party 0 has data
//    - pmask=2 (010): Only Party 1 has data
//    - pmask=3 (011): Parties 0 and 1 have data
//    - pmask=7 (111): All 3 parties have data
//
// 3. rmask in peval:
//    - Specifies which parties execute the computation
//    - Must be subset of input pmask (parties need data to compute)
//    - Example: rmask=3 means Parties 0,1 compute together
//
// 4. Uniform control flow:
//    - uniform_cond: Condition must be same across all parties
//    - uniform_while: Loop condition must be synchronized
//    - MPC protocols ensure boolean values are consistent
//    - verify_uniform=true enables runtime verification
//
// 5. Security guarantees:
//    - Individual values never revealed to other parties
//    - Computation happens on secret shares
//    - Only final results can be revealed (via conv to all parties)
//    - Intermediate values remain secret-shared throughout
