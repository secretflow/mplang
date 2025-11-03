// RUN: mpir-opt %s -split-input-file -verify-diagnostics

// Test shfl (shuffle) operation - error cases

module {
  func.func @shfl_type_mismatch(%x: !mpir.mp<tensor<10xf32>, 1>)
                                -> !mpir.mp<tensor<20xf32>, 2> {
    // Invalid: inner type changes (tensor<10xf32> -> tensor<20xf32>)
    // expected-error @+1 {{shfl does not change value type}}
    %result = mpir.shfl %x {src_ranks = array<i64: 0>}
              : !mpir.mp<tensor<10xf32>, 1>
              -> !mpir.mp<tensor<20xf32>, 2>
    return %result : !mpir.mp<tensor<20xf32>, 2>
  }
}

// -----

module {
  func.func @shfl_wrong_src_ranks_length(%x: !mpir.mp<tensor<10xf32>, 1>)
                                         -> !mpir.mp<tensor<10xf32>, 7> {
    // Invalid: output pmask=7 (3 parties), but src_ranks has only 2 elements
    // expected-error @+1 {{src_ranks length (2) must equal number of output parties (3, pmask=7)}}
    %result = mpir.shfl %x {src_ranks = array<i64: 0, 0>}
              : !mpir.mp<tensor<10xf32>, 1>
              -> !mpir.mp<tensor<10xf32>, 7>
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }
}

// -----

module {
  func.func @shfl_invalid_src_rank(%x: !mpir.mp<tensor<10xf32>, 1>)
                                   -> !mpir.mp<tensor<10xf32>, 2> {
    // Invalid: src_rank=2 (party 2) is not in input pmask=1 (party 0)
    // expected-error @+1 {{src_ranks[0] = 2 is not in input pmask 1. Source party must have data.}}
    %result = mpir.shfl %x {src_ranks = array<i64: 2>}
              : !mpir.mp<tensor<10xf32>, 1>
              -> !mpir.mp<tensor<10xf32>, 2>
    return %result : !mpir.mp<tensor<10xf32>, 2>
  }
}

// -----

module {
  func.func @shfl_out_of_range_src_rank(%x: !mpir.mp<tensor<10xf32>, 1>)
                                        -> !mpir.mp<tensor<10xf32>, 2> {
    // Invalid: src_rank=64 is out of valid range [0, 64)
    // expected-error @+1 {{src_ranks[0] = 64 is out of valid range [0, 64)}}
    %result = mpir.shfl %x {src_ranks = array<i64: 64>}
              : !mpir.mp<tensor<10xf32>, 1>
              -> !mpir.mp<tensor<10xf32>, 2>
    return %result : !mpir.mp<tensor<10xf32>, 2>
  }
}

// -----

module {
  func.func @shfl_encryption_schema_mismatch(%x: !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 1>)
                                             -> !mpir.mp<!mpir.enc<tensor<10xf32>, "ckks">, 2> {
    // Invalid: encryption schema changes (paillier -> ckks)
    // expected-error @+1 {{shfl does not change value type}}
    %result = mpir.shfl %x {src_ranks = array<i64: 0>}
              : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 1>
              -> !mpir.mp<!mpir.enc<tensor<10xf32>, "ckks">, 2>
    return %result : !mpir.mp<!mpir.enc<tensor<10xf32>, "ckks">, 2>
  }
}
