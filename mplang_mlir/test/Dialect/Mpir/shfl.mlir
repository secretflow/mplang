// RUN: mpir-opt %s -split-input-file -verify-diagnostics

// Test shfl (shuffle) operation - positive cases

module {
  // Basic shuffle: party 0 -> party 1
  func.func @shfl_basic(%x: !mpir.mp<tensor<10xf32>, 1>)
                        -> !mpir.mp<tensor<10xf32>, 2> {
    %result = mpir.shfl %x {src_ranks = array<i64: 0>}
              : !mpir.mp<tensor<10xf32>, 1>
              -> !mpir.mp<tensor<10xf32>, 2>
    return %result : !mpir.mp<tensor<10xf32>, 2>
  }

  // Broadcast: party 0 -> all parties
  func.func @shfl_broadcast(%x: !mpir.mp<tensor<10xf32>, 1>)
                            -> !mpir.mp<tensor<10xf32>, 7> {
    %result = mpir.shfl %x {src_ranks = array<i64: 0, 0, 0>}
              : !mpir.mp<tensor<10xf32>, 1>
              -> !mpir.mp<tensor<10xf32>, 7>
    // Parties 0,1,2 all pull from party 0
    return %result : !mpir.mp<tensor<10xf32>, 7>
  }

  // Shuffle encrypted data
  func.func @shfl_encrypted(%enc: !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 1>)
                            -> !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 2> {
    %result = mpir.shfl %enc {src_ranks = array<i64: 0>}
              : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 1>
              -> !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 2>
    return %result : !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 2>
  }

  // Multiple sources: parties 0,1 -> parties 2,3
  func.func @shfl_multi_source(%x0: !mpir.mp<tensor<10xf32>, 1>,
                               %x1: !mpir.mp<tensor<10xf32>, 2>)
                               -> (!mpir.mp<tensor<10xf32>, 4>,
                                   !mpir.mp<tensor<10xf32>, 8>) {
    // Merge sources first
    %merged = mpir.conv(%x0, %x1)
              : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>)
              -> !mpir.mp<tensor<10xf32>, 3>

    // Party 2 pulls from party 0
    %to_p2 = mpir.shfl %merged {src_ranks = array<i64: 0>}
             : !mpir.mp<tensor<10xf32>, 3>
             -> !mpir.mp<tensor<10xf32>, 4>

    // Party 3 pulls from party 1
    %to_p3 = mpir.shfl %merged {src_ranks = array<i64: 1>}
             : !mpir.mp<tensor<10xf32>, 3>
             -> !mpir.mp<tensor<10xf32>, 8>

    return %to_p2, %to_p3 : !mpir.mp<tensor<10xf32>, 4>,
                            !mpir.mp<tensor<10xf32>, 8>
  }

  // Secret sharing distribution pattern
  func.func @shfl_share_distribution(%share0: !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>,
                                     %share1: !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>,
                                     %share2: !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>)
                                     -> (!mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>,
                                         !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 2>,
                                         !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 4>) {
    // Party 0 keeps share0
    %s0_at_p0 = mpir.shfl %share0 {src_ranks = array<i64: 0>}
                : !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>
                -> !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>

    // Share1 goes to party 1
    %s1_at_p1 = mpir.shfl %share1 {src_ranks = array<i64: 0>}
                : !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>
                -> !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 2>

    // Share2 goes to party 2
    %s2_at_p2 = mpir.shfl %share2 {src_ranks = array<i64: 0>}
                : !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>
                -> !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 4>

    return %s0_at_p0, %s1_at_p1, %s2_at_p2
           : !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 1>,
             !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 2>,
             !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3_share">, 4>
  }
}
