cmake_minimum_required(VERSION 3.30.0)
project(micro)

set(pybind11_DIR /Users/jeffrey/miniconda3/envs/rush/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(micro micro.cpp)