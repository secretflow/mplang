// RUN: mpir-opt %s | FileCheck %s

// Example: Paillier Homomorphic Encryption Workflow
// This example demonstrates a typical PHE computation pattern:
//   1. Party 0 generates keypair (pk, sk)
//   2. Party 0 encrypts local data with pk
//   3. Party 1 encrypts local data with pk (received from Party 0)
//   4. Parties perform homomorphic addition on encrypted data
//   5. Result is sent to Party 0 for decryption

module {
  // Step 1: Key generation on Party 0
  // Generates Paillier keypair (public key, secret key)
  func.func @phe_keygen() -> (!mpir.mp<tensor<0xi8>, 1>, !mpir.mp<tensor<0xi8>, 1>) {
    // Generate public key on Party 0
    %pk = mpir.peval () {
      fn_type = "phe.keygen",
      fn_attrs = {scheme = "paillier", key_part = "public"},
      rmask = 1 : i64
    } : () -> !mpir.mp<tensor<0xi8>, 1>

    // Generate secret key on Party 0
    %sk = mpir.peval () {
      fn_type = "phe.keygen",
      fn_attrs = {scheme = "paillier", key_part = "secret"},
      rmask = 1 : i64
    } : () -> !mpir.mp<tensor<0xi8>, 1>

    return %pk, %sk : !mpir.mp<tensor<0xi8>, 1>, !mpir.mp<tensor<0xi8>, 1>
  }

  // Step 2 & 3: Encryption
  // Party 0 encrypts data x0, Party 1 encrypts data x1
  func.func @phe_encrypt(%x0: !mpir.mp<tensor<100xf32>, 1>,
                         %x1: !mpir.mp<tensor<100xf32>, 2>,
                         %pk: !mpir.mp<tensor<0xi8>, 1>)
                         -> (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>,
                             !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>) {
    // Broadcast pk to Party 1 for encryption
    // conv combines Party 0's pk into a value visible to both parties
    %pk_shared = mpir.conv(%pk) : (!mpir.mp<tensor<0xi8>, 1>)
                                -> !mpir.mp<tensor<0xi8>, 3>

    // Party 0 encrypts x0 with pk
    %enc_x0 = mpir.peval @phe_encrypt_op(%x0, %pk_shared) {
      rmask = 1 : i64
    } : (!mpir.mp<tensor<100xf32>, 1>, !mpir.mp<tensor<0xi8>, 3>)
      -> !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>

    // Party 1 encrypts x1 with pk
    %enc_x1 = mpir.peval @phe_encrypt_op(%x1, %pk_shared) {
      rmask = 2 : i64
    } : (!mpir.mp<tensor<100xf32>, 2>, !mpir.mp<tensor<0xi8>, 3>)
      -> !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>

    return %enc_x0, %enc_x1 : !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>,
                              !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>
  }

  // Step 4: Homomorphic computation
  // Add encrypted values from both parties
  func.func @phe_add(%enc_x0: !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>,
                     %enc_x1: !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>)
                     -> !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3> {
    // Combine encrypted values from both parties
    %enc_combined = mpir.conv(%enc_x0, %enc_x1)
                    : (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>,
                       !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>)
                    -> !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>

    // Perform homomorphic addition
    // Both parties can compute on encrypted data without decryption
    %result = mpir.peval @phe_add_op(%enc_combined) {
      rmask = 3 : i64
    } : (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>)
      -> !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>

    return %result : !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>
  }

  // Step 5: Decryption
  // Party 0 decrypts the final result using secret key
  func.func @phe_decrypt(%enc_result: !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>,
                         %sk: !mpir.mp<tensor<0xi8>, 1>)
                         -> !mpir.mp<tensor<100xf32>, 1> {
    // Only Party 0 performs decryption (has secret key)
    %result = mpir.peval @phe_decrypt_op(%enc_result, %sk) {
      rmask = 1 : i64
    } : (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>,
         !mpir.mp<tensor<0xi8>, 1>)
      -> !mpir.mp<tensor<100xf32>, 1>

    return %result : !mpir.mp<tensor<100xf32>, 1>
  }

  // Complete workflow: keygen -> encrypt -> compute -> decrypt
  func.func @phe_workflow(%x0: !mpir.mp<tensor<100xf32>, 1>,
                          %x1: !mpir.mp<tensor<100xf32>, 2>)
                          -> !mpir.mp<tensor<100xf32>, 1> {
    // Generate keys
    %pk, %sk = call @phe_keygen() : () -> (!mpir.mp<tensor<0xi8>, 1>, !mpir.mp<tensor<0xi8>, 1>)

    // Encrypt inputs
    %enc_x0, %enc_x1 = call @phe_encrypt(%x0, %x1, %pk)
                       : (!mpir.mp<tensor<100xf32>, 1>,
                          !mpir.mp<tensor<100xf32>, 2>,
                          !mpir.mp<tensor<0xi8>, 1>)
                       -> (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>,
                           !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>)

    // Compute on encrypted data
    %enc_result = call @phe_add(%enc_x0, %enc_x1)
                  : (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 1>,
                     !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 2>)
                  -> !mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>

    // Decrypt result
    %result = call @phe_decrypt(%enc_result, %sk)
              : (!mpir.mp<!mpir.encrypted<tensor<100xf32>, paillier>, 3>,
                 !mpir.mp<tensor<0xi8>, 1>)
              -> !mpir.mp<tensor<100xf32>, 1>

    return %result : !mpir.mp<tensor<100xf32>, 1>
  }
}

// Key concepts demonstrated:
// 1. rmask usage: Controls which parties execute operations
//    - rmask=1: Party 0 only (keygen, decryption)
//    - rmask=2: Party 1 only (encryption)
//    - rmask=3: Both parties (homomorphic computation)
//
// 2. conv operation: Combines data from different parties
//    - Used to share public key with Party 1
//    - Used to combine encrypted values for joint computation
//
// 3. Type system: MP<Encrypted<T, schema>, pmask>
//    - Inner type: Encrypted<tensor<100xf32>, paillier>
//    - Outer type: MP<..., pmask> tracks party ownership
//
// 4. Security: Secret key never leaves Party 0
//    - Only pk is shared via conv
//    - Computation happens on encrypted data
//    - Only Party 0 can decrypt final result
