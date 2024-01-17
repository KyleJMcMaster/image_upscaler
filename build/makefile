OBJECTS = build/main.o build/Image.o build/ColorChanger.o build/CUDAColorChanger.o build/CUDAKernels.o

all: $(OBJECTS)
	nvcc $(OBJECTS) -o img_up

build/main.o: src/main.cpp
	g++ -c $< -o $@

build/%.o: src/%.cu headers/%.cuh
	nvcc -c $< -o $@

build/%.o: src/%.cpp headers/%.h
	g++ -c $< -o $@

