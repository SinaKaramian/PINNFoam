#include <gtest/gtest.h>
#include "fvCFD.H"

TEST(OpenFOAM, CompilesLinksAndRuns)
{
  Foam::scalar a = 3.25;
  Foam::scalar b = 4.75;
  Foam::scalar c = a + b;

  EXPECT_NEAR(static_cast<double>(c), 8.0, 1e-12);
}
