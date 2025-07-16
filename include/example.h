#pragma once
#include "cxx-test/src/main.rs.h"
#include "rust/cxx.h"
#include <iostream>

rust::Vec<Shared> f(rust::Vec<Shared> elements);

/*
void stupid() {
    std::cout << "you are not stupid" << std::endl;
}
*/