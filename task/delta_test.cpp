//
// Created by 1 on 19.02.2022.
//

#include "FCascade.h"

extern bool verbose;

//#include "data/big_delta.h"

int main()
{
  verbose = true;

  vector<int8_t> input
      = {99, 1, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98,
         98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 91};
  {
    printf("--------------------Delta------------------------\n");

    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
  {
    printf("--------------------Delta OVERFLOW_DELTA_FOR_INTERVAL------------------------\n");

    struct CompResult res;
    nvcompCascadedFormatOpts opts = {0,1,
                                     0, nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::NORMAL_DELTA};
    opts.delta_opts.delta_mode = nvcompCascadedFormatOpts::DeltaOpts::DeltaMode::OVERFLOW_DELTA_FOR_INTERVAL;
    nvcompCascadedFormatOpts::DeltaOpts::DeltaMode t;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0, &opts));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
}