CMAKE_MINIMUM_REQUIRED(VERSION 3.7.1)

include(CMakePrintHelpers)

PROJECT(pfem2dealii DESCRIPTION "The pfem2dealii library")
SET(PROJECT_NAME "pfem2dealii")
SET(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE(${PROJECT_PATH}/cmake/setup_deal_ii.cmake)

file(GLOB src_files ${PROJECT_PATH}/src/*.cpp)
file(GLOB header_files ${PROJECT_PATH}/src/*.h)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_PATH}/install" CACHE PATH "..." FORCE)
endif()

add_library(pfem2dealii SHARED ${src_files})

target_include_directories(pfem2dealii SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})
target_link_libraries(pfem2dealii PUBLIC ${DEAL_II_LIBRARIES})

add_subdirectory(utils/repairMesh)
add_subdirectory(utils/repairMesh3D)

install(TARGETS pfem2dealii)
install(FILES ${header_files} DESTINATION include)

set_target_properties(pfem2dealii PROPERTIES VERSION 1.0)
set_target_properties(pfem2dealii PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

MESSAGE("The ${PROJECT_NAME} library will be installed to ${CMAKE_INSTALL_PREFIX}")
cmake_print_variables(CMAKE_BUILD_TYPE)
