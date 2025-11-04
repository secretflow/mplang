#!/bin/bash
# Development environment setup for MPLang + MLIR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up MPLang development environment..."
echo "   Repository: $REPO_ROOT"

# Add mplang to PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
echo "✓ Added mplang to PYTHONPATH"

# Check for MLIR build
MLIR_BUILD="$REPO_ROOT/mplang_mlir/build/python_packages/mplang_mlir"
if [ -d "$MLIR_BUILD" ]; then
    export PYTHONPATH="$MLIR_BUILD:$PYTHONPATH"
    echo "✓ Found MLIR bindings at: mplang_mlir/build/python_packages/mplang_mlir"
else
    echo "⚠️  MLIR bindings not built yet"
    echo ""
    echo "To build MLIR backend, run:"
    echo "  cd $REPO_ROOT/mplang_mlir"
    echo "  mkdir -p build && cd build"
    echo "  cmake -G Ninja .. \\"
    echo "    -DCMAKE_BUILD_TYPE=Debug \\"
    echo "    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \\"
    echo "    -DMLIR_ENABLE_BINDINGS_PYTHON=ON"
    echo "  ninja"
    echo ""
fi

# Verify imports
echo ""
echo "🧪 Testing imports..."
python3 -c "
import sys
success = True

try:
    import mplang
    print('✓ mplang')
except ImportError as e:
    print(f'✗ mplang: {e}')
    success = False

try:
    import mplang_mlir
    print('✓ mplang_mlir')
except ImportError as e:
    print(f'⚠ mplang_mlir: {e}')
    print('  (Build C++ first if needed)')

if success:
    print('')
    print('✅ Development environment ready!')
    print('')
    print('Quick commands:')
    print('  pytest tests/              # Run tests')
    print('  python tutorials/0_basic.py # Run tutorial')
    print('  uv run pytest              # Run with uv')
else:
    sys.exit(1)
"

# Save environment for subshells
cat > "$REPO_ROOT/.env.dev" << EOF
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHONPATH="$MLIR_BUILD:$PYTHONPATH"
EOF

echo ""
echo "💡 To use in new shells, run:"
echo "   source $REPO_ROOT/mplang_mlir/scripts/dev-env.sh"
echo ""
echo "Or add to your .bashrc/.zshrc:"
echo "   alias mplang-dev='source $REPO_ROOT/mplang_mlir/scripts/dev-env.sh'"
