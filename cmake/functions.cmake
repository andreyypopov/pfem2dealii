# additional functions for setup of the pfem2dealii library and simulation cases

macro(setup_cuda target_name)
	set(CMAKE_CUDA_ARCHITECTURES 61 70 75)
	enable_language(CUDA)
	#find_package(CUDAToolkit)
	#include_directories(${CUDAToolkit_INCLUDE_DIRS})
	#link_directories(${CUDAToolkit_LIBRARY_DIR})
	#target_link_libraries(${target_name} PRIVATE CUDA::cudart_static CUDA::cuda_driver)
	set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 61 70 75)
	set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	add_definitions(-DUSE_CUDA)
endmacro()
