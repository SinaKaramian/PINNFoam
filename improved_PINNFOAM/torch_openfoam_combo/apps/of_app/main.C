#include <torch/torch.h>

// STL
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>

// OpenFOAM
#include "fvCFD.H"

using namespace Foam;
using namespace torch::indexing;

static inline double clampMin(double v, double mn) { return v < mn ? mn : v; }

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"

    runTime.setTime(1.0, 1);

    #include "createMesh.H"

    volScalarField vf
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    volScalarField vf_nn(vf);
    vf_nn.rename("vf_nn");
    vf_nn.writeOpt(IOobject::AUTO_WRITE);
    vf_nn.primitiveFieldRef() = 0.0;
    vf_nn.correctBoundaryConditions();

    volScalarField error_c
    (
        IOobject
        (
            "error_c",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", vf.dimensions(), 0.0)
    );

    // ------------------------------------------------------------------ //
    // Hyperparameters

    DynamicList<label> hiddenLayers;
    scalar optimizerStep;
    label maxIterations;

    scalar lambdaData;
    scalar lambdaPDE;
    scalar lambdaBC;

    label warmupEpochs;
    label pdeRampEpochs;

    scalar trainFraction;
    scalar pdeFraction;
    label seed;

    const fvSolution& fvSolutionDict(mesh);
    const dictionary& aiDict = fvSolutionDict.subDict("AI");

    hiddenLayers  = aiDict.get<DynamicList<label>>("hiddenLayers");
    optimizerStep = aiDict.get<scalar>("optimizerStep");
    maxIterations = aiDict.get<label>("maxIterations");

    lambdaData    = aiDict.get<scalar>("lambdaData");
    lambdaPDE     = aiDict.get<scalar>("lambdaPDE");
    lambdaBC      = aiDict.get<scalar>("lambdaBC");

    warmupEpochs  = aiDict.get<label>("warmupEpochs");
    pdeRampEpochs = aiDict.get<label>("pdeRampEpochs");

    trainFraction = aiDict.get<scalar>("trainFraction");
    pdeFraction   = aiDict.get<scalar>("pdeFraction");
    seed          = aiDict.get<label>("seed");

    trainFraction = min(max(trainFraction, scalar(0)), scalar(1));
    pdeFraction   = min(max(pdeFraction,   scalar(0)), scalar(1));
    warmupEpochs  = max(warmupEpochs,  label(0));
    pdeRampEpochs = max(pdeRampEpochs, label(0));

    if (optimizerStep <= VSMALL)
    {
        WarningInFunction
            << "optimizerStep is <= 0 (=" << optimizerStep
            << "). Overriding to 1e-3." << nl << endl;
        optimizerStep = 1e-3;
    }

    // Old libtorch API in your build expects TypeMeta
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));

    // ---------------- GPU/CPU device selection ----------------
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA, 0);

        // IMPORTANT FIX: device_count() may be int8_t (signed char) -> cast to int for OpenFOAM Info<<
        Info<< "Torch CUDA found. device_count="
            << static_cast<int>(torch::cuda::device_count())
            << " -> using cuda:0" << nl << endl;
    }
    else
    {
        Info<< "Torch CUDA NOT found -> using CPU" << nl << endl;
    }

    Info<< "Training setup:" << nl
        << "  timeDir        = " << runTime.timeName() << nl
        << "  nCells         = " << mesh.nCells() << nl
        << "  hiddenLayers   = " << hiddenLayers << nl
        << "  optimizerStep  = " << optimizerStep << nl
        << "  trainFraction  = " << trainFraction << nl
        << "  pdeFraction    = " << pdeFraction << nl
        << "  seed           = " << seed << nl
        << "  maxIterations  = " << maxIterations << nl
        << "  lambdaData     = " << lambdaData << nl
        << "  lambdaPDE      = " << lambdaPDE << nl
        << "  lambdaBC       = " << lambdaBC << nl
        << "  warmupEpochs   = " << warmupEpochs << nl
        << "  pdeRampEpochs  = " << pdeRampEpochs << nl << endl;

    // ------------------------------------------------------------------ //
    // Build internal training tensors (2D: x,y)

    const label N = mesh.nCells();

    const scalarField& T_if = vf.internalField();
    const vectorField& C_if = mesh.C().internalField();

    // Create on CPU first (accessor writes require CPU), then move to selected device
    auto cpu_opts = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

    torch::Tensor T_cpu  = torch::empty({N, 1}, cpu_opts);
    torch::Tensor XY_cpu = torch::empty({N, 2}, cpu_opts);

    {
        auto Ta = T_cpu.accessor<double,2>();
        auto Xa = XY_cpu.accessor<double,2>();
        forAll(T_if, i)
        {
            Ta[i][0] = static_cast<double>(T_if[i]);
            Xa[i][0] = static_cast<double>(C_if[i].x());
            Xa[i][1] = static_cast<double>(C_if[i].y());
        }
    }

    // Move to device (CPU or CUDA)
    torch::Tensor T_tensor  = T_cpu.to(device);
    torch::Tensor XY_tensor = XY_cpu.to(device);

    // Compute scaling for inputs: x' = 2*(x-xmin)/(xmax-xmin) - 1
    torch::Tensor xy_min = std::get<0>(XY_tensor.min(0, /*keepdim=*/true));
    torch::Tensor xy_max = std::get<0>(XY_tensor.max(0, /*keepdim=*/true));
    torch::Tensor denom  = (xy_max - xy_min).clamp_min(1e-12);
    torch::Tensor XY_scaled = 2.0*(XY_tensor - xy_min)/denom - 1.0;

    // Physical derivative scaling factors: d/dx = sx * d/dx', with sx=2/(xmax-xmin)
    torch::Tensor s = 2.0/denom; // shape (1,2)
    const double sx = s.index({0,0}).item<double>();
    const double sy = s.index({0,1}).item<double>();

    // Output scaling statistics (for numerical conditioning)
    torch::Tensor T_mean = T_tensor.mean();
    torch::Tensor T_std  = T_tensor.std().clamp_min(1e-12);

    // Train network to output scaled value; compute losses in physical units.
    torch::Tensor T_scaled_target = (T_tensor - T_mean)/T_std;

    // ------------------------------------------------------------------ //
    // Subsampling

    trainFraction = std::min(1.0, std::max(0.0, trainFraction));
    pdeFraction   = std::min(1.0, std::max(0.0, pdeFraction));

    const label nTrain = std::max<label>(1, label(std::round(trainFraction * N)));
    const label nPDE   = std::max<label>(1, label(std::round(pdeFraction   * N)));

    torch::manual_seed(seed);

    torch::Tensor perm = torch::randperm(
        N,
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );

    torch::Tensor idxTrain = perm.narrow(/*dim=*/0, /*start=*/0,      /*length=*/nTrain);
    torch::Tensor idxPDE   = perm.narrow(/*dim=*/0, /*start=*/N-nPDE, /*length=*/nPDE);

    torch::Tensor XY_data = XY_scaled.index_select(0, idxTrain).clone();
    torch::Tensor T_data  = T_tensor .index_select(0, idxTrain).clone();
    XY_data.set_requires_grad(false);

    torch::Tensor XY_pde  = XY_scaled.index_select(0, idxPDE).clone();
    XY_pde.set_requires_grad(true);

    Info<< "Subsampling:" << nl
        << "  trainFraction = " << trainFraction << "  (nTrain=" << nTrain << ")" << nl
        << "  pdeFraction   = " << pdeFraction   << "  (nPDE="   << nPDE   << ")" << nl
        << "  seed          = " << seed << nl << endl;

    // ------------------------------------------------------------------ //
    // Boundary tensors

    auto makePatchXY = [&](const fvPatch& pp) -> torch::Tensor
    {
        const label nF = pp.size();

        torch::Tensor XYp_cpu = torch::empty({nF, 2}, cpu_opts);
        auto acc = XYp_cpu.accessor<double,2>();
        const vectorField& Cf = pp.Cf();
        for (label i = 0; i < nF; ++i)
        {
            acc[i][0] = static_cast<double>(Cf[i].x());
            acc[i][1] = static_cast<double>(Cf[i].y());
        }

        torch::Tensor XYp = XYp_cpu.to(device);
        return 2.0*(XYp - xy_min)/denom - 1.0;
    };

    auto makePatchN = [&](const fvPatch& pp) -> torch::Tensor
    {
        const label nF = pp.size();

        torch::Tensor Np_cpu = torch::empty({nF, 2}, cpu_opts);
        auto acc = Np_cpu.accessor<double,2>();
        tmp<vectorField> tnf = pp.nf();
        const vectorField& nf = tnf();
        for (label i = 0; i < nF; ++i)
        {
            acc[i][0] = static_cast<double>(nf[i].x());
            acc[i][1] = static_cast<double>(nf[i].y());
        }
        return Np_cpu.to(device);
    };

    const polyBoundaryMesh& pbm = mesh.boundaryMesh();

    auto patchId = [&](const word& name) -> label
    {
        forAll(pbm, i)
        {
            if (pbm[i].name() == name) return i;
        }
        return -1;
    };

    const label idWest  = patchId("west");
    const label idEast  = patchId("east");
    const label idSouth = patchId("south");
    const label idNorth = patchId("north");

    torch::Tensor XY_w, N_w, XY_e, N_e, XY_s, N_s, XY_n;
    bool hasW=false, hasE=false, hasS=false, hasN=false;

    if (idWest >= 0)
    {
        const fvPatch& pp = mesh.boundary()[idWest];
        XY_w = makePatchXY(pp); XY_w.set_requires_grad(true);
        N_w  = makePatchN(pp);
        hasW = true;
    }
    if (idEast >= 0)
    {
        const fvPatch& pp = mesh.boundary()[idEast];
        XY_e = makePatchXY(pp); XY_e.set_requires_grad(true);
        N_e  = makePatchN(pp);
        hasE = true;
    }
    if (idSouth >= 0)
    {
        const fvPatch& pp = mesh.boundary()[idSouth];
        XY_s = makePatchXY(pp); XY_s.set_requires_grad(true);
        N_s  = makePatchN(pp);
        hasS = true;
    }
    if (idNorth >= 0)
    {
        const fvPatch& pp = mesh.boundary()[idNorth];
        XY_n = makePatchXY(pp); XY_n.set_requires_grad(true);
        hasN = true;
    }

    if (!hasW || !hasE || !hasS || !hasN)
    {
        WarningInFunction
            << "One or more expected patches not found. "
            << "Found: west=" << hasW << ", east=" << hasE
            << ", south=" << hasS << ", north=" << hasN << nl << endl;
    }

    const double T_north = 100.0;
    const double g_west  = 500.0;
    const double g_zero  = 0.0;

    // ------------------------------------------------------------------ //
    // Build MLP (input 2 -> output 1 (scaled))

    torch::nn::Sequential nn;

    nn->push_back(torch::nn::Linear(2, hiddenLayers[0]));
    nn->push_back(torch::nn::Tanh());

    for (label L = 1; L < hiddenLayers.size(); ++L)
    {
        nn->push_back(torch::nn::Linear(hiddenLayers[L-1], hiddenLayers[L]));
        nn->push_back(torch::nn::Tanh());
    }

    nn->push_back(torch::nn::Linear(hiddenLayers.back(), 1));

    nn->to(device);
    nn->to(torch::kDouble);

    torch::optim::Adam optimizer(
        nn->parameters(),
        torch::optim::AdamOptions(static_cast<double>(optimizerStep))
    );

    const int PRINT_EVERY = 2000;

    // ------------------------------------------------------------------ //
    // Training loop

    for (label epoch = 1; epoch <= maxIterations; ++epoch)
    {
        optimizer.zero_grad();

        torch::Tensor T_pred_data_scaled = nn->forward(XY_data);
        torch::Tensor T_pred_data = T_pred_data_scaled*T_std + T_mean;

        torch::Tensor mse_data = torch::mse_loss(T_pred_data, T_data);

        torch::Tensor mse_pde = torch::zeros({}, mse_data.options());

        double lambdaPDE_eff = lambdaPDE;
        if (epoch <= warmupEpochs)
        {
            lambdaPDE_eff = 0.0;
        }
        else if (pdeRampEpochs > 0 && epoch <= warmupEpochs + pdeRampEpochs)
        {
            const double t = double(epoch - warmupEpochs)/double(pdeRampEpochs);
            lambdaPDE_eff = lambdaPDE * t;
        }

        if (lambdaPDE_eff > 0.0)
        {
            torch::Tensor T_pred_pde_scaled = nn->forward(XY_pde);
            torch::Tensor T_pred_pde = T_pred_pde_scaled*T_std + T_mean;

            auto grad1 = torch::autograd::grad(
                {T_pred_pde},
                {XY_pde},
                {torch::ones_like(T_pred_pde)},
                /*retain_graph=*/true,
                /*create_graph=*/true
            )[0];

            auto dTdxp = grad1.index({Slice(), 0}).view({-1,1});
            auto dTdyp = grad1.index({Slice(), 1}).view({-1,1});

            auto grad2x = torch::autograd::grad(
                {dTdxp}, {XY_pde}, {torch::ones_like(dTdxp)}, /*retain=*/true, /*create=*/true
            )[0].index({Slice(), 0}).view({-1,1});

            auto grad2y = torch::autograd::grad(
                {dTdyp}, {XY_pde}, {torch::ones_like(dTdyp)}, /*retain=*/true, /*create=*/true
            )[0].index({Slice(), 1}).view({-1,1});

            torch::Tensor lap = (sx*sx)*grad2x + (sy*sy)*grad2y;

            mse_pde = torch::mse_loss(lap, torch::zeros_like(lap));
        }

        torch::Tensor mse_bc = torch::zeros({}, mse_data.options());

        if (lambdaBC > 0.0)
        {
            torch::Tensor bcAccum = torch::zeros({}, mse_data.options());
            int bcTerms = 0;

            if (hasN)
            {
                torch::Tensor Tn_scaled = nn->forward(XY_n);
                torch::Tensor Tn = Tn_scaled*T_std + T_mean;
                bcAccum = bcAccum + torch::mse_loss(Tn, torch::full_like(Tn, T_north));
                bcTerms++;
            }

            auto bcNeumann = [&](const torch::Tensor& XYp, const torch::Tensor& Np, double gTarget)
            {
                torch::Tensor Tp_scaled = nn->forward(XYp);
                torch::Tensor Tp = Tp_scaled*T_std + T_mean;

                auto g1 = torch::autograd::grad(
                    {Tp}, {XYp}, {torch::ones_like(Tp)}, /*retain=*/true, /*create=*/true
                )[0];

                torch::Tensor dTdx = sx * g1.index({Slice(),0});
                torch::Tensor dTdy = sy * g1.index({Slice(),1});

                torch::Tensor flux =
                    Np.index({Slice(),0})*dTdx +
                    Np.index({Slice(),1})*dTdy;

                torch::Tensor target = torch::full_like(flux, gTarget);
                return torch::mse_loss(flux, target);
            };

            if (hasW)
            {
                bcAccum = bcAccum + bcNeumann(XY_w, N_w, g_west);
                bcTerms++;
            }
            if (hasE)
            {
                bcAccum = bcAccum + bcNeumann(XY_e, N_e, g_zero);
                bcTerms++;
            }
            if (hasS)
            {
                bcAccum = bcAccum + bcNeumann(XY_s, N_s, g_zero);
                bcTerms++;
            }

            if (bcTerms > 0) mse_bc = bcAccum / double(bcTerms);
        }

        torch::Tensor loss = lambdaData*mse_data + lambdaPDE_eff*mse_pde + lambdaBC*mse_bc;

        loss.backward();
        optimizer.step();

        if (epoch == 1 || epoch % PRINT_EVERY == 0 || epoch == maxIterations)
        {
            double gradNorm2 = 0.0;
            for (const auto& p : nn->parameters())
            {
                if (p.grad().defined())
                {
                    gradNorm2 += p.grad().pow(2).sum().item<double>();
                }
            }

            Info<< "Epoch = " << epoch << nl
                << "Data MSE = " << mse_data.item<double>() << nl
                << "PDE  MSE = " << mse_pde.item<double>() << nl
                << "BC   MSE = " << mse_bc.item<double>() << nl
                << "lambdaPDE_eff = " << lambdaPDE_eff << nl
                << "Training MSE = " << loss.item<double>() << nl
                << "GradNorm = " << Foam::sqrt(gradNorm2) << nl
                << endl;
        }
    }

    // ------------------------------------------------------------------ //
    // Write vf_nn and error fields

    torch::NoGradGuard noGrad;

    torch::Tensor T_all_scaled = nn->forward(XY_scaled);
    torch::Tensor T_all = T_all_scaled*T_std + T_mean;
    torch::Tensor T_flat = T_all.view({-1});

    torch::Tensor T_flat_cpu = T_flat.to(torch::kCPU).contiguous();
    auto Tacc = T_flat_cpu.accessor<double,1>();

    forAll(vf_nn, cellI)
    {
        vf_nn[cellI] = Tacc[cellI];
    }
    vf_nn.correctBoundaryConditions();

    error_c = Foam::mag(vf - vf_nn);

    const scalarField diff = mag(vf.internalField() - vf_nn.internalField());
    scalar errInf  = gMax(diff);
    scalar errMean = gAverage(diff);

    Info<< "max(|internal field - internal field_nn|) = " << errInf << nl
        << "mean(|internal field - internal field_nn|) = " << errMean << nl << endl;

    volVectorField vf_grad ("vf_grad", fvc::grad(vf));
    volVectorField vf_nn_grad ("vf_nn_grad", fvc::grad(vf_nn));
    volScalarField error_grad_c ("error_grad_c", Foam::mag(vf_grad - vf_nn_grad));

    error_c.write();
    vf_nn.write();
    vf_nn_grad.write();
    vf_grad.write();
    error_grad_c.write();

    Info<< "End\n" << endl;
    return 0;
}

