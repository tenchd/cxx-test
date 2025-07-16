
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

  std::vector<int> stdv_value;
  for (auto i: stdv) {
    stdv_value.push_back(i.v);
  }

  writeVectorToFile2(stdv_value, "output.txt");

  // write back into rust::Vec and return
  rust::Vec<Shared> output;
  for (auto i: stdv) {
    output.push_back(i);
  }

  //stupidconnorlala();

  return output;
}