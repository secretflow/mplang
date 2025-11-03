//===- MpirTypes.td -------------------------------------------*- tablegen -*-===//
// Mpir type ODS declarations.
//
// Type System Architecture:
//   Layer 1: Security/Crypto Types
//     - Encrypted<T, schema, attrs>: Encrypted wrapper around any type
//
//   Layer 2: Multi-Party Distribution
//     - MP<T, pmask>: Multi-party distributed value (static pmask)
//     - MPDynamic<T>: Multi-party value with runtime-determined pmask
//
// Key Design Decisions:
//   - Use MLIR builtin types (tensor, tuple) for data representation
//   - Encrypted<T> preserves T's structure (allows homomorphic ops)
//   - MP<T> wraps any type to add distribution semantics
//   - Composition: MP<Encrypted<tensor>> = encrypted tensor distributed across parties
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "MpirDialect.td"

//===----------------------------------------------------------------------===//
// Layer 1: Encoding Types
//===----------------------------------------------------------------------===//

def MPIR_Encoded : TypeDef<Mpir_Dialect, "Encoded"> {
  let mnemonic = "enc";
  let summary = "Encoded data with transformation schema";
  let description = [{
    Represents encoded/transformed data that preserves the structure of the inner type
    while applying an encoding transformation.

    The inner type can be any MLIR type (typically tensor or tuple).
    The schema specifies the encoding method, which can be:
    - Encryption: "paillier", "ckks", "bfv", "aes-gcm", etc.
    - Secret Sharing: "aby3", "replicated", "additive", "shamir", etc.
    - Garbled Circuits: "garbled_circuit", "yao", etc.
    - Plain (identity): "plain" (explicitly unencoded)

    Optional encoding_attrs store scheme-specific parameters (key sizes, scales, ring size, threshold, etc.).

    Key insight: Encoded<T, schema> is still "T-like" - it supports operations on T
    under the encoding (e.g., homomorphic ops on encrypted data, MPC ops on secret shares).

    Examples:
      !mpir.enc<tensor<10xf32>, "paillier">                    // PHE encryption
      !mpir.enc<tensor<10xf32>, "paillier", {key_size = 2048}> // PHE with params
      !mpir.enc<tensor<10xf32>, "ckks", {scale = 40, level = 5}> // FHE
      !mpir.enc<tensor<10xf32>, "aby3">                        // ABY3 secret sharing
      !mpir.enc<tensor<10xf32>, "additive">                    // Additive shares
      !mpir.enc<tensor<10xf32>, "shamir", {threshold = 2, n = 5}> // Shamir secret sharing
      !mpir.enc<tuple<f32, i64>, "garbled_circuit">           // Garbled circuit
      !mpir.enc<tensor<10xf32>, "plain">                      // Explicitly plain
  }];

  let parameters = (ins
    "::mlir::Type":$innerType,
    "::std::string":$schema,
    OptionalParameter<"::mlir::DictionaryAttr">:$encodingAttrs
  );

  let assemblyFormat = "`<` $innerType `,` $schema (`,` $encodingAttrs^)? `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$innerType,
      "::llvm::StringRef":$schema
    ), [{
      return $_get(innerType.getContext(), innerType, schema.str(), nullptr);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$innerType,
      "::llvm::StringRef":$schema,
      "::mlir::DictionaryAttr":$encodingAttrs
    ), [{
      return $_get(innerType.getContext(), innerType, schema.str(), encodingAttrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Check if this encodes a tensor type
    bool isTensorEncoded() const {
      return ::mlir::isa<::mlir::TensorType>(getInnerType());
    }

    /// Check if this encodes a tuple type
    bool isTupleEncoded() const {
      return ::mlir::isa<::mlir::TupleType>(getInnerType());
    }

    /// Get the underlying tensor type if this encodes a tensor
    ::mlir::TensorType getTensorType() const {
      return ::mlir::dyn_cast<::mlir::TensorType>(getInnerType());
    }

    /// Get the underlying tuple type if this encodes a tuple
    ::mlir::TupleType getTupleType() const {
      return ::mlir::dyn_cast<::mlir::TupleType>(getInnerType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Layer 2: Multi-Party Distribution
//===----------------------------------------------------------------------===//

def MPIR_MP : TypeDef<Mpir_Dialect, "MP"> {
  let mnemonic = "mp";
  let summary = "Multi-party distributed value with static pmask";
  let description = [{
    Represents a value distributed across multiple parties in SPMD computation.

    The inner type can be:
    - MLIR builtin types (tensor, tuple, etc.)
    - Encoded types (!mpir.enc<...>) for encrypted/secret-shared data
    - Any other type that makes sense in multi-party context

    The pmask (party mask) is a 64-bit integer where bit i indicates party i
    holds this value.

    Design Philosophy:
    - pmask is a SEMANTIC property: "who holds this data"
    - Distinct from execution mask (rmask): "who executes this operation"
    - Type composition enables: MP<Encoded<Tensor>> = encoded tensor held by subset of parties

    Examples:
      !mpir.mp<tensor<10xf32>, 7>                          // Plaintext on parties 0,1,2
      !mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 1>   // Encrypted on party 0
      !mpir.mp<!mpir.enc<tensor<10xf32>, "aby3">, 7>       // Secret shares on parties 0,1,2
      !mpir.mp<tuple<f32, i64>, 3>                         // Tuple on parties 0,1
      !mpir.mp<!mpir.enc<tuple<f32, i64>, "additive">, 3>  // Additive shares on parties 0,1
  }];

  let parameters = (ins
    "::mlir::Type":$innerType,
    "uint64_t":$pmask
  );

  let assemblyFormat = "`<` $innerType `,` $pmask `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$innerType,
      "uint64_t":$pmask
    ), [{
      return $_get(innerType.getContext(), innerType, pmask);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Check if a specific party holds this value (bit i in pmask)
    bool hasParty(unsigned partyId) const {
      if (partyId >= 64)
        return false;
      return (getPmask() & (1ULL << partyId)) != 0;
    }

    /// Get list of party IDs that hold this value
    ::llvm::SmallVector<unsigned> getPartyList() const {
      ::llvm::SmallVector<unsigned> parties;
      uint64_t mask = getPmask();
      for (unsigned i = 0; i < 64; ++i) {
        if (mask & (1ULL << i))
          parties.push_back(i);
      }
      return parties;
    }

    /// Count number of parties holding this value
    unsigned getNumParties() const {
      return __builtin_popcountll(getPmask());
    }

    /// Check if inner type is a tensor (direct or through encoding)
    bool isTensor() const {
      if (::mlir::isa<::mlir::TensorType>(getInnerType()))
        return true;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.isTensorEncoded();
      return false;
    }

    /// Check if inner type is a tuple (direct or through encoding)
    bool isTuple() const {
      if (::mlir::isa<::mlir::TupleType>(getInnerType()))
        return true;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.isTupleEncoded();
      return false;
    }

    /// Check if inner type is encoded
    bool isEncoded() const {
      return ::mlir::isa<::mlir::mpir::EncodedType>(getInnerType());
    }

    /// Get the underlying tensor type (works through encoded wrapper)
    ::mlir::TensorType getTensorType() const {
      if (auto tensor = ::mlir::dyn_cast<::mlir::TensorType>(getInnerType()))
        return tensor;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.getTensorType();
      return nullptr;
    }

    /// Get the underlying tuple type (works through encoded wrapper)
    ::mlir::TupleType getTupleType() const {
      if (auto tuple = ::mlir::dyn_cast<::mlir::TupleType>(getInnerType()))
        return tuple;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.getTupleType();
      return nullptr;
    }

    /// Get encoding schema if inner type is encoded
    ::std::optional<::llvm::StringRef> getEncodingSchema() const {
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.getSchema();
      return ::std::nullopt;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Dynamic pmask variant (for runtime-determined distribution)
//===----------------------------------------------------------------------===//

def MPIR_MPDynamic : TypeDef<Mpir_Dialect, "MPDynamic"> {
  let mnemonic = "mp_dynamic";
  let summary = "Multi-party value with runtime-determined pmask";
  let description = [{
    Similar to MP type but the pmask is determined at runtime.
    Used for operations where the output distribution depends on runtime values
    (e.g., reading from file where success depends on file system state).

    This avoids the ambiguity of using a special pmask value like 0xFFFFFFFFFFFFFFFF
    which could be confused with "all 64 parties hold the value".

    Example:
      !mpir.mp_dynamic<tensor<10xf32>>
      !mpir.mp_dynamic<!mpir.enc<tensor<10xf32>, "ckks">>
  }];

  let parameters = (ins "::mlir::Type":$innerType);
  let assemblyFormat = "`<` $innerType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$innerType), [{
      return $_get(innerType.getContext(), innerType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Check if inner type is a tensor
    bool isTensor() const {
      if (::mlir::isa<::mlir::TensorType>(getInnerType()))
        return true;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.isTensorEncoded();
      return false;
    }

    /// Check if inner type is encoded
    bool isEncoded() const {
      return ::mlir::isa<::mlir::mpir::EncodedType>(getInnerType());
    }

    /// Get the underlying tensor type (works through encoded wrapper)
    ::mlir::TensorType getTensorType() const {
      if (auto tensor = ::mlir::dyn_cast<::mlir::TensorType>(getInnerType()))
        return tensor;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncodedType>(getInnerType()))
        return enc.getTensorType();
      return nullptr;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints for Ops (Type Aliases)
//===----------------------------------------------------------------------===//

// Base type constraints
def MPIR_EncodedType : Type<CPred<"::llvm::isa<::mlir::mpir::EncodedType>($_self)">,
                             "Encoded type", "::mlir::mpir::EncodedType">;

def MPIR_MPType : Type<CPred<"::llvm::isa<::mlir::mpir::MPType>($_self)">,
                       "Multi-party type (static pmask)", "::mlir::mpir::MPType">;

def MPIR_MPDynamicType : Type<CPred<"::llvm::isa<::mlir::mpir::MPDynamicType>($_self)">,
                               "Multi-party type (dynamic pmask)", "::mlir::mpir::MPDynamicType">;

// Any MP type (static or dynamic pmask)
def MPIR_AnyMPType : AnyTypeOf<[MPIR_MPType, MPIR_MPDynamicType],
                                "Multi-party distributed type">;

//===----------------------------------------------------------------------===//
// Common Type Patterns (Convenient Aliases)
//===----------------------------------------------------------------------===//

// Pattern: MP<Tensor> - plaintext tensor distributed across parties
def MPIR_MPTensor : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTensor() && "
        "!::llvm::cast<::mlir::mpir::MPType>($_self).isEncoded()">,
  "Multi-party plaintext tensor",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Encoded<Tensor>> - encoded tensor distributed across parties (METensor)
def MPIR_METensor : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTensor() && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isEncoded()">,
  "Multi-party encoded tensor",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Tensor> | MP<Encoded<Tensor>> - any tensor (plain or encoded)
def MPIR_MPAnyTensor : AnyTypeOf<[MPIR_MPTensor, MPIR_METensor],
                                 "Multi-party tensor (plaintext or encoded)">;

// Pattern: MP<Tuple> - plaintext table distributed across parties
def MPIR_MPTable : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTuple() && "
        "!::llvm::cast<::mlir::mpir::MPType>($_self).isEncoded()">,
  "Multi-party plaintext table",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Encoded<Tuple>> - encoded table distributed across parties
def MPIR_METable : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTuple() && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isEncoded()">,
  "Multi-party encoded table",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Tuple> | MP<Encoded<Tuple>> - any table (plain or encoded)
def MPIR_MPAnyTable : AnyTypeOf<[MPIR_MPTable, MPIR_METable],
                                "Multi-party table (plaintext or encoded)">;

// Pattern: MP<Encoded<*>> - any encoded multi-party value
def MPIR_MPEncoded : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isEncoded()">,
  "Multi-party encoded value",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<*> where * is Tensor or Tuple (plain or encrypted)
def MPIR_MPTensorOrTable : AnyTypeOf<[MPIR_MPAnyTensor, MPIR_MPAnyTable],
                                     "Multi-party tensor or table">;

//===----------------------------------------------------------------------===//
// Dynamic Variants
//===----------------------------------------------------------------------===//

// Pattern: MPDynamic<Tensor>
def MPIR_MPDynamicTensor : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPDynamicType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPDynamicType>($_self).isTensor()">,
  "Multi-party tensor with dynamic pmask",
  "::mlir::mpir::MPDynamicType"
>;

// Pattern: MP<Tensor> | MPDynamic<Tensor> (static or dynamic)
def MPIR_MPAnyTensorStaticOrDynamic : AnyTypeOf<[MPIR_MPAnyTensor, MPIR_MPDynamicTensor],
                                                 "Multi-party tensor (static or dynamic pmask)">;
