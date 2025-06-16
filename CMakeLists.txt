cmake_minimum_required(VERSION 3.10)
project(play_hackrf CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(HACKRF REQUIRED libhackrf)


add_executable(play_hackrf play.cpp)

target_include_directories(play_hackrf PRIVATE ${HACKRF_INCLUDE_DIRS})
target_link_libraries(play_hackrf PRIVATE ${HACKRF_LIBRARIES})