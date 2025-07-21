#pragma once
#include "cxx-test/src/main.rs.h"
#include "rust/cxx.h"
#include <iostream>
#include "cxx-test/include/custom_cg.hpp"


rust::Vec<Shared> f(rust::Vec<Shared> elements);

FlattenedVec2 go(FlattenedVec2 shared_jl_cols);

rust::Vec<size_t> no_sharing(rust::Vec<size_t> elements);

/*
void stupidconnorlalala() {
    std::cout << "you are not stupid" << std::endl;
}
*/

