CMAKE_MINIMUM_REQUIRED(VERSION 3.7.1)

include(CMakePrintHelpers)

SET(PROJECT_NAME "TurekBenchmark")
SET(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

PROJECT(${PROJECT_NAME})

INCLUDE(${PROJECT_PATH}/../../cmake/setup_deal_ii.cmake)
INCLUDE(${PROJECT_PATH}/../../cmake/functions.cmake)

SET_IF_EMPTY(PFEM2DEALII_DIR "${PROJECT_PATH}/../../install")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cu)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${DEAL_II_INCLUDE_DIRS} ${PFEM2DEALII_DIR}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${PFEM2DEALII_DIR}/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE pfem2dealii ${DEAL_II_LIBRARIES})

setup_cuda(${PROJECT_NAME})
