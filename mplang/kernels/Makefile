# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CXX = g++
# -march=native enables PCLMULQDQ if the host CPU supports it.
# -mpclmul -maes are explicit flags if native doesn't pick them up, but native is safer for local dev.
CXXFLAGS = -O3 -Wall -shared -fPIC -march=native -mpclmul -maes -fopenmp

TARGET = libmplang_kernels.so
SRCS = gf128.cpp okvs.cpp okvs_opt.cpp ldpc.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) $(OBJS)
