#pragma once
#include "cxx-test/src/main.rs.h"
#include "rust/cxx.h"
#include <iostream>
#include "cxx-test/include/custom_cg.hpp"


//rust::Vec<Shared> f(rust::Vec<Shared> elements);

FlattenedVec go(FlattenedVec shared_jl_cols);

void sprs_test(rust::Vec<size_t> rust_col_ptrs, rust::Vec<size_t> rust_row_indices, rust::Vec<double> rust_values);

/*
void stupidconnorlalala() {
    std::cout << "you are not stupid" << std::endl;
}
*/

