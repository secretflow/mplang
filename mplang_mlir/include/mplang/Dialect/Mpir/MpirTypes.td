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
// Layer 1: Security/Crypto Types
//===----------------------------------------------------------------------===//

def MPIR_Encrypted : TypeDef<Mpir_Dialect, "Encrypted"> {
  let mnemonic = "encrypted";
  let summary = "Encrypted data with cryptographic schema";
  let description = [{
    Represents encrypted data that preserves the structure of the inner type
    while adding cryptographic properties.

    The inner type can be any MLIR type (typically tensor or tuple).
    The schema specifies the encryption algorithm (e.g., "paillier", "ckks", "aes-gcm").
    Optional crypto_attrs store algorithm-specific parameters (key sizes, scales, etc.).

    Key insight: Encrypted<T> is still "T-like" - it supports operations on T
    (e.g., homomorphic addition on Encrypted<tensor>).

    Examples:
      !mpir.encrypted<tensor<10xf32>, "paillier">
      !mpir.encrypted<tensor<10xf32>, "paillier", {key_size = 2048}>
      !mpir.encrypted<tensor<10xf32>, "ckks", {scale = 40, level = 5}>
      !mpir.encrypted<tuple<f32, i64>, "fpe">  // Format-preserving encryption
  }];

  let parameters = (ins
    "::mlir::Type":$innerType,
    "::std::string":$schema,
    OptionalParameter<"::mlir::DictionaryAttr">:$cryptoAttrs
  );

  let assemblyFormat = "`<` $innerType `,` $schema (`,` $cryptoAttrs^)? `>`";

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
      "::mlir::DictionaryAttr":$cryptoAttrs
    ), [{
      return $_get(innerType.getContext(), innerType, schema.str(), cryptoAttrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Check if this encrypts a tensor type
    bool isTensorEncrypted() const {
      return ::mlir::isa<::mlir::TensorType>(getInnerType());
    }

    /// Check if this encrypts a tuple type
    bool isTupleEncrypted() const {
      return ::mlir::isa<::mlir::TupleType>(getInnerType());
    }

    /// Get the underlying tensor type if this encrypts a tensor
    ::mlir::TensorType getTensorType() const {
      return ::mlir::dyn_cast<::mlir::TensorType>(getInnerType());
    }

    /// Get the underlying tuple type if this encrypts a tuple
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
    - Encrypted types (!mpir.encrypted<...>)
    - Any other type that makes sense in multi-party context

    The pmask (party mask) is a 64-bit integer where bit i indicates party i
    holds this value.

    Design Philosophy:
    - pmask is a SEMANTIC property: "who holds this data"
    - Distinct from execution mask (rmask): "who executes this operation"
    - Type composition enables: MP<Encrypted<Tensor>> = encrypted tensor held by subset of parties

    Examples:
      !mpir.mp<tensor<10xf32>, 7>                               // Plaintext on parties 0,1,2
      !mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 1>  // Ciphertext on party 0
      !mpir.mp<tuple<f32, i64>, 3>                              // Tuple on parties 0,1
      !mpir.mp<!mpir.encrypted<tuple<f32, i64>, "fpe">, 15>     // Encrypted tuple on parties 0-3
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

    /// Check if inner type is a tensor (direct or through encryption)
    bool isTensor() const {
      if (::mlir::isa<::mlir::TensorType>(getInnerType()))
        return true;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
        return enc.isTensorEncrypted();
      return false;
    }

    /// Check if inner type is a tuple (direct or through encryption)
    bool isTuple() const {
      if (::mlir::isa<::mlir::TupleType>(getInnerType()))
        return true;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
        return enc.isTupleEncrypted();
      return false;
    }

    /// Check if inner type is encrypted
    bool isEncrypted() const {
      return ::mlir::isa<::mlir::mpir::EncryptedType>(getInnerType());
    }

    /// Get the underlying tensor type (works through encrypted wrapper)
    ::mlir::TensorType getTensorType() const {
      if (auto tensor = ::mlir::dyn_cast<::mlir::TensorType>(getInnerType()))
        return tensor;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
        return enc.getTensorType();
      return nullptr;
    }

    /// Get the underlying tuple type (works through encrypted wrapper)
    ::mlir::TupleType getTupleType() const {
      if (auto tuple = ::mlir::dyn_cast<::mlir::TupleType>(getInnerType()))
        return tuple;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
        return enc.getTupleType();
      return nullptr;
    }

    /// Get encryption schema if inner type is encrypted
    ::std::optional<::llvm::StringRef> getEncryptionSchema() const {
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
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
      !mpir.mp_dynamic<!mpir.encrypted<tensor<10xf32>, "ckks">>
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
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
        return enc.isTensorEncrypted();
      return false;
    }

    /// Check if inner type is encrypted
    bool isEncrypted() const {
      return ::mlir::isa<::mlir::mpir::EncryptedType>(getInnerType());
    }

    /// Get the underlying tensor type (works through encrypted wrapper)
    ::mlir::TensorType getTensorType() const {
      if (auto tensor = ::mlir::dyn_cast<::mlir::TensorType>(getInnerType()))
        return tensor;
      if (auto enc = ::mlir::dyn_cast<::mlir::mpir::EncryptedType>(getInnerType()))
        return enc.getTensorType();
      return nullptr;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints for Ops (Type Aliases)
//===----------------------------------------------------------------------===//

// Base type constraints
def MPIR_EncryptedType : Type<CPred<"::llvm::isa<::mlir::mpir::EncryptedType>($_self)">,
                               "Encrypted type", "::mlir::mpir::EncryptedType">;

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
        "!::llvm::cast<::mlir::mpir::MPType>($_self).isEncrypted()">,
  "Multi-party plaintext tensor",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Encrypted<Tensor>> - encrypted tensor distributed across parties (METensor)
def MPIR_METensor : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTensor() && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isEncrypted()">,
  "Multi-party encrypted tensor",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Tensor> | MP<Encrypted<Tensor>> - any tensor (plain or encrypted)
def MPIR_MPAnyTensor : AnyTypeOf<[MPIR_MPTensor, MPIR_METensor],
                                 "Multi-party tensor (plaintext or encrypted)">;

// Pattern: MP<Tuple> - plaintext table distributed across parties
def MPIR_MPTable : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTuple() && "
        "!::llvm::cast<::mlir::mpir::MPType>($_self).isEncrypted()">,
  "Multi-party plaintext table",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Encrypted<Tuple>> - encrypted table distributed across parties
def MPIR_METable : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isTuple() && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isEncrypted()">,
  "Multi-party encrypted table",
  "::mlir::mpir::MPType"
>;

// Pattern: MP<Tuple> | MP<Encrypted<Tuple>> - any table (plain or encrypted)
def MPIR_MPAnyTable : AnyTypeOf<[MPIR_MPTable, MPIR_METable],
                                "Multi-party table (plaintext or encrypted)">;

// Pattern: MP<Encrypted<*>> - any encrypted multi-party value
def MPIR_MPEncrypted : Type<
  CPred<"::llvm::isa<::mlir::mpir::MPType>($_self) && "
        "::llvm::cast<::mlir::mpir::MPType>($_self).isEncrypted()">,
  "Multi-party encrypted value",
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
