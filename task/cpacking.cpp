
#include "FCascade.h"

bool verbose = false;

int main()
{
  vector<uint8_t> input = { 96,97,97,97,97,97,97,97,98,98,98,98,98,98,98,98,98,98,98,98,98,99,99,99,99,99,99,99,99,99,99,99,99 };
  {
    printf("--------------------Delta------------------------\n");

    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0));
    FCascade::show_stat(input, res, true, true);

    printf("--------------------------------------------\n");
  }
  {
    printf("\n--------------------Delta + BP------------------------\n");
    // vector<uint8_t> input = {3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 6,
    //  6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9};
    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 1));
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
    printf("\n--------------------Delta > BP------------------------\n");
    // vector<uint8_t> input = {3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 6,
    //                          6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9};
    struct CompResult res;
    REQUIRE(FCascade::cascade(input, res, 0, 1, 0));
    FCascade::show_stat(input, res);
    struct CompResult res1;
    REQUIRE(FCascade::cascade(res.output, res1, 0, 0, 1));
    FCascade::show_stat(res.output, res1);
  }

  {
    #include "pb99.h"
    printf("\n--------------------BP 99------------------------\n");
    struct CompResult res;

    REQUIRE(FCascade::cascade(input, res, 0, 0, 1));
//    FCascade::show_stat(input, res);
    FCascade::short_stat(input, res);

  }


  {
#include "pb66.h"
    printf("\n--------------------BP 66------------------------\n");
    struct CompResult res;

    REQUIRE(FCascade::cascade(input, res, 0, 0, 1));
    //    FCascade::show_stat(input, res);
    FCascade::short_stat(input, res);

  }

    {
  #include "pb96.h"
      printf("\n--------------------BP 96------------------------\n");
      struct CompResult res;

      REQUIRE(FCascade::cascade(input, res, 0, 0, 1));
      //    FCascade::show_stat(input, res);
      FCascade::short_stat(input, res);

    }


    {
#include "pb96-1.h"
      printf("\n--------------------BP 96 1------------------------\n");
      struct CompResult res;

      REQUIRE(FCascade::cascade(input, res, 0, 0, 1));
      //    FCascade::show_stat(input, res);
      FCascade::short_stat(input, res);

    }

  printf("\n\ndone\n");
}
