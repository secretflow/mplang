CXX = g++
# -march=native enables PCLMULQDQ if the host CPU supports it.
# -mpclmul -maes are explicit flags if native doesn't pick them up, but native is safer for local dev.
CXXFLAGS = -O3 -Wall -shared -fPIC -march=native -mpclmul -maes

TARGET = libmplang_kernels.so
SRCS = gf128.cpp okvs.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) $(OBJS)
