#include "fvCFD.H"

int main(int argc, char** argv)
{
  Foam::scalar a = 1.0;
  Foam::scalar b = 2.0;
  Foam::scalar c = a + b;

  Foam::Info << "OpenFOAM scalar smoke: " << a << " + " << b << " = " << c << Foam::endl;

  return (Foam::mag(c - Foam::scalar(3.0)) < Foam::SMALL) ? 0 : 1;
}
