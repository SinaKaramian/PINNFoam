#include "fvCFD.H"
#include "OSspecific.H"   // mkDir(), isFile()

using namespace Foam;


// Read DT from constant/physicalProperties or constant/transportProperties.
static dimensionedScalar readDT(const Time& runTime)
{
    // Thermal conductivity dimensions: [1 1 -3 -1 0 0 0]
    const dimensionSet dimKappa(1, 1, -3, -1, 0, 0, 0);
    const dimensionedScalar DTdefault("DT", dimKappa, 1.0);

    const fileName phys  = runTime.constant()/"physicalProperties";

    IOdictionary physicalProperties
    (
        IOobject
        (
            "physicalProperties",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    dimensionedScalar DT(physicalProperties.lookup("DT"));
    return DT;
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    Info<< "Reading field T" << nl << endl;

    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    const dimensionedScalar DT = readDT(runTime);

    Info<< "Starting laplacianMatrixFoam" << nl << endl;

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Assemble Laplace system: div(DT grad T) = 0
        fvScalarMatrix TEqn(fvm::laplacian(DT, T));


        // Solve
        Info<< "Solving TEqn ..." << nl << endl;
        TEqn.solve();

        runTime.write();
        Info<< "End of time step " << runTime.timeName() << nl << endl;
    }

    Info<< "End" << nl << endl;
    return 0;
}

