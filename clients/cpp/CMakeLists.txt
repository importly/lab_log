cmake_minimum_required(VERSION 3.10)
project(lab_log_cpp VERSION 0.1.0)

add_library(lab_log INTERFACE)
target_include_directories(lab_log INTERFACE include)