//
// Created by 1 on 19.02.2022.
//

#include "FCascade.h"

bool verbose = true;

#include "data/big_delta.h"

int main()
{
//  vector<uint8_t> input
//      = {96, 97, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98,
//         98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};
  {
    printf("--------------------Delta------------------------\n");

    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
}