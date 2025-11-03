//===- MpirTypes.td -------------------------------------------*- tablegen -*-===//
// Mpir type ODS declarations.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "MpirDialect.td"

//===----------------------------------------------------------------------===//
// TensorType: multi-party tensor with pmask
//===----------------------------------------------------------------------===//

def MPIR_TensorType : TypeDef<Mpir_Dialect, "Tensor"> {
  let mnemonic = "tensor";
  let summary = "Multi-party tensor with shape, element type, and party mask";
  let description = [{
    A tensor type representing multi-party distributed data with:
    - shape: Static shape dimensions (ArrayRef<int64_t>)
    - elementType: MLIR builtin type (e.g., f32, i64, complex<f32>)
    - pmask: Party mask as uint64_t bit vector
      * Bit i set means party i holds this tensor
      * Example: pmask=7 (0b111) means parties 0,1,2 hold the data

    This type encodes the core semantic of SPMD multi-party computation:
    "what the data is" (shape, dtype) AND "who holds it" (pmask).

    Examples:
      !mpir.tensor<10xf32, 7>           // Shape [10], f32, parties 0,1,2
      !mpir.tensor<3x4xi64, 1>          // Shape [3,4], i64, only party 0
      !mpir.tensor<f32, 3>              // Scalar f32, parties 0,1
      !mpir.tensor<5x10xcomplex<f64>, 15> // Complex tensor, parties 0,1,2,3
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "tensor shape">:$shape,
    "::mlir::Type":$elementType,
    "uint64_t":$pmask
  );

  let assemblyFormat = "`<` custom<TensorShape>($shape) $elementType `,` $pmask `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "::mlir::Type":$elementType,
      "uint64_t":$pmask
    ), [{
      return $_get(elementType.getContext(), shape, elementType, pmask);
    }]>
  ];

  let extraClassDeclaration = [{
    // Utility methods for accessing tensor properties

    /// Get the number of dimensions
    unsigned getRank() const { return getShape().size(); }

    /// Check if this is a scalar (rank 0)
    bool isScalar() const { return getRank() == 0; }

    /// Get number of elements (product of shape dimensions)
    /// Returns -1 if any dimension is dynamic (though we don't support dynamic shapes yet)
    int64_t getNumElements() const {
      auto shape = getShape();
      if (shape.empty())
        return 1; // scalar
      int64_t result = 1;
      for (auto dim : shape) {
        if (dim < 0)
          return -1; // dynamic dimension
        result *= dim;
      }
      return result;
    }

    /// Check if a specific party holds this tensor (bit i in pmask)
    bool hasParty(unsigned partyId) const {
      if (partyId >= 64)
        return false;
      return (getPmask() & (1ULL << partyId)) != 0;
    }

    /// Get list of party IDs that hold this tensor
    ::llvm::SmallVector<unsigned> getPartyList() const {
      ::llvm::SmallVector<unsigned> parties;
      uint64_t mask = getPmask();
      for (unsigned i = 0; i < 64; ++i) {
        if (mask & (1ULL << i))
          parties.push_back(i);
      }
      return parties;
    }

    /// Count number of parties holding this tensor
    unsigned getNumParties() const {
      return __builtin_popcountll(getPmask());
    }
  }];
}

//===----------------------------------------------------------------------===//
// TableType: multi-party table with pmask
//===----------------------------------------------------------------------===//

def MPIR_TableType : TypeDef<Mpir_Dialect, "Table"> {
  let mnemonic = "table";
  let summary = "Multi-party table with schema and party mask";
  let description = [{
    A table type representing structured data (like DataFrame) with:
    - schema: Tuple type containing column types
    - pmask: Party mask indicating which parties hold this table

    Tables are used for structured/heterogeneous data where different columns
    can have different types (including string, date, etc.).

    Examples:
      !mpir.table<tuple<f32, i64, f32>, 7>  // 3 columns, parties 0,1,2
      !mpir.table<tuple<i32>, 1>            // Single column, party 0 only
  }];

  let parameters = (ins
    "::mlir::TupleType":$schema,
    "uint64_t":$pmask
  );

  let assemblyFormat = "`<` $schema `,` $pmask `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::TupleType":$schema,
      "uint64_t":$pmask
    ), [{
      return $_get(schema.getContext(), schema, pmask);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get number of columns
    unsigned getNumColumns() const {
      return getSchema().size();
    }

    /// Get type of column at index
    ::mlir::Type getColumnType(unsigned index) const {
      return getSchema().getType(index);
    }

    /// Check if a specific party holds this table
    bool hasParty(unsigned partyId) const {
      if (partyId >= 64)
        return false;
      return (getPmask() & (1ULL << partyId)) != 0;
    }

    /// Get list of party IDs that hold this table
    ::llvm::SmallVector<unsigned> getPartyList() const {
      ::llvm::SmallVector<unsigned> parties;
      uint64_t mask = getPmask();
      for (unsigned i = 0; i < 64; ++i) {
        if (mask & (1ULL << i))
          parties.push_back(i);
      }
      return parties;
    }

    /// Count number of parties holding this table
    unsigned getNumParties() const {
      return __builtin_popcountll(getPmask());
    }
  }];
}
