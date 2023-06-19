FIND_PACKAGE(deal.II 9.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II version 9.4 or higher. ***\n"
    "Either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

MESSAGE("Found ${DEAL_II_PACKAGE_NAME} library version ${DEAL_II_PACKAGE_VERSION} at ${DEAL_II_PATH}")

SET_IF_EMPTY(CMAKE_BUILD_TYPE "Release")

SET(CMAKE_BUILD_TYPE ${DEAL_II_BUILD_TYPE} CACHE STRING
  "Choose the type of build, options are: Debug, Release or DebugRelease"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
