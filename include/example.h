#pragma once
#include "cxx-test/src/main.rs.h"
#include "rust/cxx.h"
#include <iostream>
#include "cxx-test/include/custom_cg.hpp"


rust::Vec<Shared> f(rust::Vec<Shared> elements);

void go(rust::Vec<rust::Vec<Shared>> shared_jl_cols);

/*
void stupidconnorlalala() {
    std::cout << "you are not stupid" << std::endl;
}
*/

