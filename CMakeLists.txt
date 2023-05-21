CMAKE_MINIMUM_REQUIRED(VERSION 3.7.1)

FIND_PACKAGE(deal.II 9.4 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

SET(CMAKE_BUILD_TYPE ${DEAL_II_BUILD_TYPE} CACHE STRING
  "Choose the type of build, options are: Debug, Release or DebugRelease"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(pfem2dealii DESCRIPTION "The pfem2dealii library")
SET(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB src_files ${PROJECT_PATH}/src/*.cpp)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_PATH}/install" CACHE PATH "..." FORCE)
endif()

add_library(pfem2dealii SHARED ${src_files})

target_include_directories(pfem2dealii SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS})
target_link_libraries(pfem2dealii PRIVATE ${DEAL_II_LIBRARIES})

install(TARGETS pfem2dealii)

set_target_properties(pfem2dealii PROPERTIES VERSION 1.0)
set_target_properties(pfem2dealii PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

MESSAGE(STATUS "Library will be installed to ${CMAKE_INSTALL_PREFIX}")
