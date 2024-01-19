OBJECTS = build/main.o build/Image.o build/ColorChanger.o build/CUDAColorChanger.o build/CUDAKernels.o build/Kernel.o

all: $(OBJECTS)
	nvcc $(OBJECTS) -o img_up -lfftw3

build/main.o: src/main.cpp
	g++ -c $< -o $@

build/%.o: src/%.cu headers/%.cuh
	nvcc -c $< -o $@

build/%.o: src/%.cpp headers/%.h
	g++ -c $< -o $@ 

