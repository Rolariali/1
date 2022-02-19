


#include "FCascade.h"

bool verbose = false;

int main()
{
//  vector<uint8_t> input = {1,1,1,1, 2, 2, 2, 3, 4, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8};

#include "data/rlebp.h"
  {
    printf("--------------------RLE------------------------\n");

    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 1, 0, 0));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
  {
    printf("\n--------------------RLE + BP------------------------\n");
    // vector<uint8_t> input = {3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 6,
    //  6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9};
    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 1, 0, 1));
    FCascade::show_stat(input, res);
  }
  {
    printf("\n--------------------BP------------------------\n");
    // vector<uint8_t> input = {3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 6,
    //                          6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9};
    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 0, 1));
    FCascade::show_stat(input, res);

  }

  {
    printf("\n--------------------RLE > BP------------------------\n");
    // vector<uint8_t> input = {3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 6,
    //                          6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9};
    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 1, 0, 0));
    FCascade::show_stat(input, res);
    struct CompResult res1;
    REQUIRE(FCascade::cascade(res.output, res1, 0, 0, 1));
    FCascade::show_stat(res.output, res1);
  }

}