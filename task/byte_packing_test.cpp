//
// Created by 1 on 21.02.2022.
//

#include "FCascade.h"

//bool verbose = true;

int main()
{
  vector<uint8_t> input
      = {96, 97, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98,
         98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};
  {
    printf("--------------------BP------------------------\n");

    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 0, 1));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
}