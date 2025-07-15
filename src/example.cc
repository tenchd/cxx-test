
#include "cxx-test/include/example.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <vector>

void increment_all_values(std::vector<Shared> &stdv) {
  for (auto i = 0; i < 3; i++) {
    stdv[i].v++;
  }
}


rust::Vec<Shared> f(rust::Vec<Shared> v) {
  for (auto shared : v) {
    std::cout << shared.v << std::endl;
  }

  // Copy the elements to a C++ std::vector using STL algorithm.
  std::vector<Shared> stdv;
  std::copy(v.begin(), v.end(), std::back_inserter(stdv));
  assert(v.size() == stdv.size());
  increment_all_values(stdv);
  for (auto i: stdv) {
    std::cout << i.v << std::endl;
  }
  // write back into rust::Vec and return
  rust::Vec<Shared> output;
  for (auto i: stdv) {
    output.push_back(i);
  }
  return output;
}