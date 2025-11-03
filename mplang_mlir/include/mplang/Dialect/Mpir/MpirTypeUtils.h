//===- MpirTypeUtils.h - Mpir Type Utility Functions ------------*- C++ -*-===//
//
// Utility functions for Mpir type system operations.
// Used by verifiers and other analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPIR_MPIRTYPEUTILS_H
#define MPLANG_DIALECT_MPIR_MPIRTYPEUTILS_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>
#include <cstdint>

namespace mlir {
namespace mpir {

// Forward declarations of generated types
class MPType;
class MPDynamicType;
class EncryptedType;

//===----------------------------------------------------------------------===//
// Pmask Extraction and Manipulation
//===----------------------------------------------------------------------===//

/// Extract pmask from MP<T, pmask> type.
/// Returns std::nullopt if type is not MPType.
std::optional<int64_t> extractPmask(Type type);

/// Extract pmask from any MP-wrapped type (handles nested MP types).
/// For MPDynamic, returns std::nullopt (dynamic pmask).
std::optional<int64_t> extractPmaskRecursive(Type type);

/// Check if two pmasks are disjoint (no overlapping bits).
/// pmask1 & pmask2 == 0
bool arePmasksDisjoint(int64_t pmask1, int64_t pmask2);

/// Check if all pmasks in the array are pairwise disjoint.
bool arePmasksDisjoint(llvm::ArrayRef<int64_t> pmasks);

/// Compute union of pmasks (bitwise OR).
int64_t computeUnionPmask(llvm::ArrayRef<int64_t> pmasks);

/// Check if pmask1 is a subset of pmask2.
/// (pmask1 & pmask2) == pmask1
bool isPmaskSubset(int64_t pmask1, int64_t pmask2);

/// Count number of parties in pmask (popcount).
int64_t countParties(int64_t pmask);

//===----------------------------------------------------------------------===//
// Type Checking Predicates
//===----------------------------------------------------------------------===//

/// Check if type is scalar boolean (i1).
bool isScalarBoolean(Type type);

/// Check if type is MP<i1, pmask> (scalar boolean with pmask).
bool isMPScalarBoolean(Type type);

/// Check if type is tensor<i1> (boolean tensor).
bool isBooleanTensor(Type type);

/// Check if type is MPType.
bool isMPType(Type type);

/// Check if type is MPDynamicType.
bool isMPDynamicType(Type type);

/// Check if type is EncryptedType.
bool isEncryptedType(Type type);

/// Extract inner type from MP<T, pmask> or MPDynamic<T>.
/// Returns the T part, or Type() if not an MP type.
Type extractInnerType(Type type);

/// Check if two types have the same "shape" (ignoring pmask).
/// For MP types: compares inner types.
/// For tensors: compares element type and shape.
/// For tuples (tables): recursively compares element types.
bool haveSameShape(Type type1, Type type2);

//===----------------------------------------------------------------------===//
// Region and Terminator Utilities
//===----------------------------------------------------------------------===//

/// Get the return types from a region's return operations.
/// Assumes all return ops in the region return the same types.
/// Returns empty array if region is empty or has no returns.
llvm::SmallVector<Type> getRegionReturnTypes(Region &region);

/// Check if region has the specified terminator type.
template <typename TerminatorOp>
bool hasTerminator(Region &region);

} // namespace mpir
} // namespace mlir

#endif // MPLANG_DIALECT_MPIR_MPIRTYPEUTILS_H
