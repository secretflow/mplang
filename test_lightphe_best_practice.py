#!/usr/bin/env python3

from lightphe import LightPHE

# 测试最佳的lightPHE使用方式
print("=== 测试lightPHE最佳实践 ===")

# 测试单个值加密
phe = LightPHE(algorithm_name="Paillier", key_size=1024, precision=6)

# 测试各种数据类型
test_cases = [
    ("int32 scalar", 42),
    ("float32 scalar", 3.14),
    ("negative int", -50),
    ("zero", 0),
    ("small float", 0.1),
]

print("\n--- 单个值测试 ---")
for name, value in test_cases:
    try:
        # 测试是否需要转换为特定类型
        if isinstance(value, int):
            # 对于整数，直接使用
            encrypted = phe.encrypt(value)
            decrypted = phe.decrypt(encrypted)
            print(f"{name}: {value} -> {decrypted} (exact: {value == decrypted})")
        else:
            # 对于浮点数，检查精度
            encrypted = phe.encrypt(value)
            decrypted = phe.decrypt(encrypted)
            
            # 检查返回类型和处理方式
            if isinstance(decrypted, list) and len(decrypted) == 1:
                decrypted = decrypted[0]
            
            diff = abs(value - decrypted) if isinstance(decrypted, (int, float)) else float('inf')
            print(f"{name}: {value} -> {decrypted} (diff: {diff})")
            
    except Exception as e:
        print(f"{name}: ERROR - {e}")

# 测试数组
print("\n--- 数组测试 ---")
array_tests = [
    ("int array", [1, 2, 3, 4, 5]),
    ("float array", [1.1, 2.2, 3.3]),
    ("mixed precision", [3.14159, 2.71828, 1.41421]),
    ("negative values", [-1, -2.5, -3.14]),
]

for name, values in array_tests:
    try:
        encrypted = phe.encrypt(values)
        decrypted = phe.decrypt(encrypted)
        
        print(f"{name}:")
        print(f"  Original:  {values}")
        print(f"  Decrypted: {decrypted}")
        
        # 计算精度
        if len(values) == len(decrypted):
            diffs = [abs(o - d) for o, d in zip(values, decrypted)]
            max_diff = max(diffs)
            print(f"  Max diff:  {max_diff}")
        
    except Exception as e:
        print(f"{name}: ERROR - {e}")

# 测试加法
print("\n--- 同态加法测试 ---")
val1, val2 = 3.14, 2.71
try:
    encrypted1 = phe.encrypt(val1)
    encrypted2 = phe.encrypt(val2)
    
    # lightPHE的加法操作
    encrypted_sum = encrypted1 + encrypted2
    decrypted_sum = phe.decrypt(encrypted_sum)
    
    expected = val1 + val2
    actual = decrypted_sum[0] if isinstance(decrypted_sum, list) else decrypted_sum
    
    print(f"Homomorphic addition: {val1} + {val2} = {expected}")
    print(f"Encrypted result: {actual}")
    print(f"Difference: {abs(expected - actual)}")
    
except Exception as e:
    print(f"Addition error: {e}")
