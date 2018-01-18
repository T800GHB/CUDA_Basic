all:
	nvcc -o demo_cube demo_cube.cu
clean:
	rm -f demo_cube
