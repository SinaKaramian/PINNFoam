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

// ------------------------------------------------------------------ //
// Expose protected fvMatrix<scalar> boundary-completion methods
class fvScalarMatrixDebug : public fvScalarMatrix
{
public:
    explicit fvScalarMatrixDebug(const fvScalarMatrix& m)
    :
        fvScalarMatrix(m)
    {}

    void applyBoundaryDiagInPlace()
    {
        // scalar => component 0
        this->addBoundaryDiag(this->diag(), 0);
    }

    void applyBoundarySourceInPlace(const bool couples = true)
    {
        this->addBoundarySource(this->source(), couples);
    }
};

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

    // New DisPINN loss hyperparameter (with backward-compatible fallbacks)
    scalar lambdaDisPINN = 0.0;

    label warmupEpochs;        // warmup: only data loss
    label dispinnRampEpochs;   // ramp: DisPINN weight increases to full

    scalar trainFraction;
    scalar dispinnFraction;    // which cells contribute to DisPINN MSE (after residual is formed)
    label seed;

    const fvSolution& fvSolutionDict(mesh);
    const dictionary& aiDict = fvSolutionDict.subDict("AI");

    hiddenLayers  = aiDict.get<DynamicList<label>>("hiddenLayers");
    optimizerStep = aiDict.get<scalar>("optimizerStep");
    maxIterations = aiDict.get<label>("maxIterations");

    lambdaData    = aiDict.get<scalar>("lambdaData");

    // --- DisPINN weight: prefer lambdaDisPINN, else (optional) reuse lambdaPDE ---
    if (aiDict.found("lambdaDisPINN"))      lambdaDisPINN = aiDict.get<scalar>("lambdaDisPINN");
    else if (aiDict.found("lambdaPDE"))     lambdaDisPINN = aiDict.get<scalar>("lambdaPDE");   // backward compat

    // --- Warmup/Ramp: prefer dispinnRampEpochs, else reuse pdeRampEpochs ---
    warmupEpochs = (aiDict.found("warmupEpochs") ? aiDict.get<label>("warmupEpochs") : 0);

    if (aiDict.found("dispinnRampEpochs"))      dispinnRampEpochs = aiDict.get<label>("dispinnRampEpochs");
    else if (aiDict.found("pdeRampEpochs"))     dispinnRampEpochs = aiDict.get<label>("pdeRampEpochs"); // compat
    else                                        dispinnRampEpochs = 0;

    trainFraction = aiDict.get<scalar>("trainFraction");

    // DisPINN sampling: prefer dispinnFraction, else reuse pdeFraction, else default 1
    if (aiDict.found("dispinnFraction"))        dispinnFraction = aiDict.get<scalar>("dispinnFraction");
    else if (aiDict.found("pdeFraction"))       dispinnFraction = aiDict.get<scalar>("pdeFraction");     // compat
    else                                        dispinnFraction = 1.0;

    seed          = aiDict.get<label>("seed");

    trainFraction   = min(max(trainFraction,   scalar(0)), scalar(1));
    dispinnFraction = min(max(dispinnFraction, scalar(0)), scalar(1));
    warmupEpochs    = max(warmupEpochs,  label(0));
    dispinnRampEpochs = max(dispinnRampEpochs, label(0));

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

        Info<< "Torch CUDA found. device_count="
            << static_cast<int>(torch::cuda::device_count())
            << " -> using cuda:0" << nl << endl;
    }
    else
    {
        Info<< "Torch CUDA NOT found -> using CPU" << nl << endl;
    }

    Info<< "Training setup:" << nl
        << "  timeDir          = " << runTime.timeName() << nl
        << "  nCells           = " << mesh.nCells() << nl
        << "  hiddenLayers     = " << hiddenLayers << nl
        << "  optimizerStep    = " << optimizerStep << nl
        << "  trainFraction    = " << trainFraction << nl
        << "  dispinnFraction  = " << dispinnFraction << nl
        << "  seed             = " << seed << nl
        << "  maxIterations    = " << maxIterations << nl
        << "  lambdaData       = " << lambdaData << nl
        << "  lambdaDisPINN    = " << lambdaDisPINN << nl
        << "  warmupEpochs     = " << warmupEpochs << nl
        << "  dispinnRampEpochs= " << dispinnRampEpochs << nl << endl;

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

    // Output scaling statistics (for numerical conditioning)
    torch::Tensor T_mean = T_tensor.mean();
    torch::Tensor T_std  = T_tensor.std().clamp_min(1e-12);

    // Train network to output scaled value; compute losses in physical units.
    torch::Tensor T_scaled_target = (T_tensor - T_mean)/T_std;

    // ------------------------------------------------------------------ //
    // Subsampling (data + DisPINN residual cells)

    const label nTrain = std::max<label>(1, label(std::round(trainFraction * N)));
    const label nDis   = std::max<label>(1, label(std::round(dispinnFraction * N)));

    torch::manual_seed(seed);

    torch::Tensor perm = torch::randperm(
        N,
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );

    torch::Tensor idxTrain = perm.narrow(/*dim=*/0, /*start=*/0,      /*length=*/nTrain);
    torch::Tensor idxDis   = perm.narrow(/*dim=*/0, /*start=*/N-nDis, /*length=*/nDis);

    torch::Tensor XY_data = XY_scaled.index_select(0, idxTrain).clone();
    torch::Tensor T_data  = T_tensor .index_select(0, idxTrain).clone();
    XY_data.set_requires_grad(false);

    Info<< "Subsampling:" << nl
        << "  trainFraction   = " << trainFraction   << "  (nTrain=" << nTrain << ")" << nl
        << "  dispinnFraction = " << dispinnFraction << "  (nDis="   << nDis   << ")" << nl
        << "  seed            = " << seed << nl << endl;

    // ------------------------------------------------------------------ //
    // Assemble discrete operator A and RHS b for DisPINN:  r = A*x - b
    //
    // Per your request:
    //   1) build fvScalarMatrix
    //   2) apply addBoundaryDiag/addBoundarySource (via debug wrapper)
    //   3) use the same LDU structure as Amul (but implemented in Torch so gradients flow)

    fvScalarMatrix A_tmp(fvm::laplacian(vf_nn)); // adjust operator here if needed
    fvScalarMatrixDebug Aeqn(A_tmp);

    Aeqn.applyBoundaryDiagInPlace();
    Aeqn.applyBoundarySourceInPlace(true);

    const scalarField& Ad = Aeqn.diag();
    const scalarField& Au = Aeqn.upper();
    const scalarField& Al = Aeqn.lower();
    const scalarField& bF = Aeqn.source();

    const lduAddressing& addr = Aeqn.lduAddr();
    const labelUList& lowerAddr = addr.lowerAddr();
    const labelUList& upperAddr = addr.upperAddr();

    const label nIntFaces = upperAddr.size();

    Info<< "DisPINN operator (LDU):" << nl
        << "  N cells        = " << N << nl
        << "  internal faces = " << nIntFaces << nl << endl;

    // Pack A and b into Torch tensors (constant during training)
    torch::Tensor A_diag_cpu  = torch::empty({N}, cpu_opts);
    torch::Tensor A_upper_cpu = torch::empty({nIntFaces}, cpu_opts);
    torch::Tensor A_lower_cpu = torch::empty({nIntFaces}, cpu_opts);
    torch::Tensor b_cpu       = torch::empty({N}, cpu_opts);

    auto idx_opts_cpu = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::Tensor lowerAddr_cpu = torch::empty({nIntFaces}, idx_opts_cpu);
    torch::Tensor upperAddr_cpu = torch::empty({nIntFaces}, idx_opts_cpu);

    {
        auto dacc = A_diag_cpu.accessor<double,1>();
        auto bacc = b_cpu.accessor<double,1>();
        forAll(Ad, i)
        {
            dacc[i] = static_cast<double>(Ad[i]);
            bacc[i] = static_cast<double>(bF[i]);
        }

        auto uacc = A_upper_cpu.accessor<double,1>();
        auto lacc = A_lower_cpu.accessor<double,1>();
        auto loacc = lowerAddr_cpu.accessor<long,1>();
        auto upacc = upperAddr_cpu.accessor<long,1>();

        for (label f = 0; f < nIntFaces; ++f)
        {
            uacc[f]  = static_cast<double>(Au[f]);
            lacc[f]  = static_cast<double>(Al[f]);
            loacc[f] = static_cast<long>(lowerAddr[f]);
            upacc[f] = static_cast<long>(upperAddr[f]);
        }
    }

    // Move constants to compute device
    torch::Tensor A_diag  = A_diag_cpu.to(device);
    torch::Tensor A_upper = A_upper_cpu.to(device);
    torch::Tensor A_lower = A_lower_cpu.to(device);
    torch::Tensor b_vec   = b_cpu.to(device);

    torch::Tensor lowerAddr_t = lowerAddr_cpu.to(device);
    torch::Tensor upperAddr_t = upperAddr_cpu.to(device);

    // Torch equivalent of OpenFOAM lduMatrix::Amul for internal field (differentiable w.r.t x)
    auto lduAmulTorch = [&](const torch::Tensor& xIn) -> torch::Tensor
    {
        torch::Tensor x = xIn.view({-1});                 // (N)
        torch::Tensor y = A_diag * x;                     // diag contribution

        // y[lower] += upper[f] * x[upper]
        torch::Tensor xU = x.index_select(0, upperAddr_t);
        y.index_add_(0, lowerAddr_t, A_upper * xU);

        // y[upper] += lower[f] * x[lower]
        torch::Tensor xL = x.index_select(0, lowerAddr_t);
        y.index_add_(0, upperAddr_t, A_lower * xL);

        return y; // (N)
    };

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
    // Training loop (Data + DisPINN)

    for (label epoch = 1; epoch <= maxIterations; ++epoch)
    {
        optimizer.zero_grad();

        // ---- Data loss ----
        torch::Tensor T_pred_data_scaled = nn->forward(XY_data);
        torch::Tensor T_pred_data = T_pred_data_scaled*T_std + T_mean;
        torch::Tensor mse_data = torch::mse_loss(T_pred_data, T_data);

        // ---- DisPINN ramp weight ----
        double lambdaDis_eff = lambdaDisPINN;

        if (epoch <= warmupEpochs)
        {
            lambdaDis_eff = 0.0;
        }
        else if (dispinnRampEpochs > 0 && epoch <= warmupEpochs + dispinnRampEpochs)
        {
            const double t = double(epoch - warmupEpochs)/double(dispinnRampEpochs);
            lambdaDis_eff = lambdaDisPINN * t;
        }

        // ---- DisPINN loss:  || A*T_nn - b ||^2  (sampled over idxDis) ----
        torch::Tensor mse_dispinn = torch::zeros({}, mse_data.options());

        if (lambdaDis_eff > 0.0)
        {
            // Need full-field NN prediction because A*x couples neighbors
            torch::Tensor T_all_scaled = nn->forward(XY_scaled);
            torch::Tensor T_all = T_all_scaled*T_std + T_mean;          // (N,1)
            torch::Tensor x = T_all.view({-1});                         // (N)

            torch::Tensor Ax = lduAmulTorch(x);                         // (N)
            torch::Tensor r  = Ax - b_vec;                              // (N)

            torch::Tensor r_sub = r.index_select(0, idxDis);
            mse_dispinn = torch::mse_loss(r_sub, torch::zeros_like(r_sub));
        }

        // ---- Total loss (DisPINN added to data loss with a ramp) ----
        torch::Tensor loss = lambdaData*mse_data + lambdaDis_eff*mse_dispinn;

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
                << "Data MSE     = " << mse_data.item<double>() << nl
                << "DisPINN MSE  = " << mse_dispinn.item<double>() << nl
                << "lambdaDis_eff= " << lambdaDis_eff << nl
                << "Training MSE = " << loss.item<double>() << nl
                << "GradNorm     = " << Foam::sqrt(gradNorm2) << nl
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

    Info<< "max(|internal field - internal field_nn|)  = " << errInf  << nl
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

