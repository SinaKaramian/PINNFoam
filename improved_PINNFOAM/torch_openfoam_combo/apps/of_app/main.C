// ============================================================
// ONE-PLACE SWITCH: FP32 vs FP64
//   1 => FP32 (float)
//   0 => FP64 (double)
// ============================================================
#define PINN_TORCH_FP32 1

#include <torch/torch.h>

// STL
#include <algorithm>
#include <cmath>
#include <string>

// OpenFOAM
#include "fvCFD.H"

using namespace Foam;
using namespace torch::indexing;

// Map the macro to a Torch dtype + C++ scalar type
#if PINN_TORCH_FP32
    using torch_real = float;
    static constexpr auto TORCH_DTYPE = torch::kFloat;
#else
    using torch_real = double;
    static constexpr auto TORCH_DTYPE = torch::kDouble;
#endif

// Minimal wrapper to expose protected boundary-completion methods
struct fvScalarMatrixExposed : public fvScalarMatrix
{
    using fvScalarMatrix::fvScalarMatrix;
    using fvScalarMatrix::addBoundaryDiag;
    using fvScalarMatrix::addBoundarySource;
};

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    runTime.setTime(1.0, 1);
    #include "createMesh.H"

    volScalarField vf
    (
        IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    volScalarField vf_nn(vf);
    vf_nn.rename("vf_nn");
    vf_nn.writeOpt(IOobject::AUTO_WRITE);
    vf_nn.primitiveFieldRef() = 0.0;
    vf_nn.correctBoundaryConditions();

    volScalarField error_c
    (
        IOobject("error_c", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", vf.dimensions(), 0.0)
    );

    // ---------------- Hyperparameters ----------------
    DynamicList<label> hiddenLayers;
    scalar optimizerStep;
    label  maxIterations;

    scalar lambdaData;
    scalar lambdaDisPINN;

    label  warmupEpochs;
    label  dispinnRampEpochs;

    scalar trainFraction;
    scalar dispinnFraction;
    label  seed;

    const fvSolution& fvSolutionDict(mesh);
    const dictionary& aiDict = fvSolutionDict.subDict("AI");

    hiddenLayers      = aiDict.get<DynamicList<label>>("hiddenLayers");
    optimizerStep     = aiDict.get<scalar>("optimizerStep");
    maxIterations     = aiDict.get<label>("maxIterations");

    lambdaData        = aiDict.get<scalar>("lambdaData");
    lambdaDisPINN     = aiDict.found("lambdaDisPINN") ? aiDict.get<scalar>("lambdaDisPINN") : 0.0;

    warmupEpochs      = aiDict.found("warmupEpochs") ? aiDict.get<label>("warmupEpochs") : 0;
    dispinnRampEpochs = aiDict.found("dispinnRampEpochs") ? aiDict.get<label>("dispinnRampEpochs") : 0;

    trainFraction     = aiDict.get<scalar>("trainFraction");
    dispinnFraction   = aiDict.found("dispinnFraction") ? aiDict.get<scalar>("dispinnFraction") : 1.0;

    seed              = aiDict.get<label>("seed");

    trainFraction     = min(max(trainFraction, scalar(0)), scalar(1));
    dispinnFraction   = min(max(dispinnFraction, scalar(0)), scalar(1));
    warmupEpochs      = max(warmupEpochs, label(0));
    dispinnRampEpochs = max(dispinnRampEpochs, label(0));

    if (optimizerStep <= VSMALL) optimizerStep = 1e-3;

    // Old libtorch API in your build expects TypeMeta
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(TORCH_DTYPE));

    // ---------------- GPU/CPU device selection ----------------
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA, 0);
        Info<< "Torch CUDA found -> using cuda:0" << nl << endl;
    }
    else
    {
        Info<< "Torch CUDA NOT found -> using CPU" << nl << endl;
    }

    // ---------------- Build tensors from mesh ----------------
    const label N = mesh.nCells();
    const scalarField& T_if = vf.internalField();
    const vectorField& C_if = mesh.C().internalField();

    // Create on CPU first (accessor writes require CPU), then move to selected device
    auto cpu_opts = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCPU);

    torch::Tensor T_cpu  = torch::empty({N, 1}, cpu_opts);
    torch::Tensor XY_cpu = torch::empty({N, 2}, cpu_opts);

    {
        auto Ta = T_cpu.accessor<torch_real,2>();
        auto Xa = XY_cpu.accessor<torch_real,2>();
        forAll(T_if, i)
        {
            Ta[i][0] = static_cast<torch_real>(T_if[i]);
            Xa[i][0] = static_cast<torch_real>(C_if[i].x());
            Xa[i][1] = static_cast<torch_real>(C_if[i].y());
        }
    }

    // Move to device (CPU or CUDA)
    torch::Tensor T_tensor  = T_cpu.to(device);
    torch::Tensor XY_tensor = XY_cpu.to(device);

    // Compute scaling for inputs: x' = 2*(x-xmin)/(xmax-xmin) - 1
    torch::Tensor xy_min = std::get<0>(XY_tensor.min(0, /*keepdim=*/true));
    torch::Tensor xy_max = std::get<0>(XY_tensor.max(0, /*keepdim=*/true));
    torch::Tensor denom  = (xy_max - xy_min).clamp_min(static_cast<torch_real>(1e-12));
    torch::Tensor XY_scaled =
        static_cast<torch_real>(2.0) * (XY_tensor - xy_min) / denom - static_cast<torch_real>(1.0);

    // Output scaling statistics (for numerical conditioning)
    torch::Tensor T_mean = T_tensor.mean();
    torch::Tensor T_std  = T_tensor.std().clamp_min(static_cast<torch_real>(1e-12));

    // ---------------- Subsampling ----------------
    const label nTrain = std::max<label>(1, label(std::round(trainFraction   * N)));
    const label nDis   = std::max<label>(1, label(std::round(dispinnFraction * N)));

    torch::manual_seed(seed);

    torch::Tensor perm = torch::randperm(
        N,
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );

    torch::Tensor idxTrain = perm.narrow(0, 0,      nTrain);
    torch::Tensor idxDis   = perm.narrow(0, N-nDis, nDis);

    torch::Tensor XY_data = XY_scaled.index_select(0, idxTrain).clone();
    torch::Tensor T_data  = T_tensor .index_select(0, idxTrain).clone();

    // ---------------- Assemble DisPINN operator: A and b ----------------
    //
    // Use your real equation here if needed. This is a placeholder operator.
    // IMPORTANT: do NOT use (fvm::laplacian(vf_nn) == 0) because RHS int causes operator== mismatch.
    fvScalarMatrixExposed Aeqn( fvm::laplacian(vf_nn) );

    // Apply boundary completion
    Aeqn.addBoundaryDiag(Aeqn.diag(), 0);
    Aeqn.addBoundarySource(Aeqn.source(), true);

    const scalarField& Ad = Aeqn.diag();
    const scalarField& Au = Aeqn.upper();
    const scalarField& Al = Aeqn.lower();
    const scalarField& bF = Aeqn.source();

    const lduAddressing& addr = Aeqn.lduAddr();
    const labelUList& lowerAddr = addr.lowerAddr();
    const labelUList& upperAddr = addr.upperAddr();

    const label nIntFaces = upperAddr.size();

    // Pack A and b into Torch tensors (constant during training)
    torch::Tensor A_diag_cpu  = torch::empty({N}, cpu_opts);
    torch::Tensor A_upper_cpu = torch::empty({nIntFaces}, cpu_opts);
    torch::Tensor A_lower_cpu = torch::empty({nIntFaces}, cpu_opts);
    torch::Tensor b_cpu       = torch::empty({N}, cpu_opts);

    auto idx_opts_cpu = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::Tensor lowerAddr_cpu = torch::empty({nIntFaces}, idx_opts_cpu);
    torch::Tensor upperAddr_cpu = torch::empty({nIntFaces}, idx_opts_cpu);

    {
        auto dacc = A_diag_cpu.accessor<torch_real,1>();
        auto bacc = b_cpu.accessor<torch_real,1>();
        forAll(Ad, i)
        {
            dacc[i] = static_cast<torch_real>(Ad[i]);
            bacc[i] = static_cast<torch_real>(bF[i]);
        }

        auto uacc  = A_upper_cpu.accessor<torch_real,1>();
        auto lacc  = A_lower_cpu.accessor<torch_real,1>();
        auto loacc = lowerAddr_cpu.accessor<long,1>();
        auto upacc = upperAddr_cpu.accessor<long,1>();

        for (label f = 0; f < nIntFaces; ++f)
        {
            uacc[f]  = static_cast<torch_real>(Au[f]);
            lacc[f]  = static_cast<torch_real>(Al[f]);
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

    // Differentiable Torch version of internal-face Amul (LDU)
    auto lduAmulTorch = [&](const torch::Tensor& xIn) -> torch::Tensor
    {
        torch::Tensor x = xIn.view({-1});     // (N)
        torch::Tensor y = A_diag * x;         // diag

        // y[lower] += upper[f] * x[upper]
        torch::Tensor xU = x.index_select(0, upperAddr_t);
        y.index_add_(0, lowerAddr_t, A_upper * xU);

        // y[upper] += lower[f] * x[lower]
        torch::Tensor xL = x.index_select(0, lowerAddr_t);
        y.index_add_(0, upperAddr_t, A_lower * xL);

        return y; // (N)
    };

    // ---------------- Build NN ----------------
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
    nn->to(TORCH_DTYPE);

    torch::optim::Adam optimizer(
        nn->parameters(),
        torch::optim::AdamOptions(static_cast<double>(optimizerStep))
    );

    const int PRINT_EVERY = 2000;

    // ---------------- Training loop ----------------
    for (label epoch = 1; epoch <= maxIterations; ++epoch)
    {
        optimizer.zero_grad();

        // Data loss
        torch::Tensor T_pred_data_scaled = nn->forward(XY_data);
        torch::Tensor T_pred_data = T_pred_data_scaled*T_std + T_mean;
        torch::Tensor mse_data = torch::mse_loss(T_pred_data, T_data);

        // DisPINN ramp
        double wDis = lambdaDisPINN;
        if (epoch <= warmupEpochs) wDis = 0.0;
        else if (dispinnRampEpochs > 0 && epoch <= warmupEpochs + dispinnRampEpochs)
        {
            const double t = double(epoch - warmupEpochs)/double(dispinnRampEpochs);
            wDis = lambdaDisPINN * t;
        }

        // DisPINN loss: ||A*T_nn - b||^2 (sample over idxDis)
        torch::Tensor mse_dispinn = torch::zeros({}, mse_data.options());
        if (wDis > 0.0)
        {
            torch::Tensor T_all_scaled = nn->forward(XY_scaled);      // (N,1)
            torch::Tensor T_all = T_all_scaled*T_std + T_mean;        // (N,1)
            torch::Tensor x = T_all.view({-1});                       // (N)

            torch::Tensor r = lduAmulTorch(x) - b_vec;                // (N)
            torch::Tensor r_sub = r.index_select(0, idxDis);

            mse_dispinn = torch::mse_loss(r_sub, torch::zeros_like(r_sub));
        }

        torch::Tensor loss = lambdaData*mse_data + wDis*mse_dispinn;
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
                << "Data MSE    = " << mse_data.item<double>() << nl
                << "DisPINN MSE = " << mse_dispinn.item<double>() << nl
                << "wDis        = " << wDis << nl
                << "Loss        = " << loss.item<double>() << nl
                << "GradNorm    = " << Foam::sqrt(gradNorm2) << nl
                << endl;
        }
    }

    // ---------------- Write vf_nn and errors ----------------
    torch::NoGradGuard noGrad;

    torch::Tensor T_all_scaled = nn->forward(XY_scaled);
    torch::Tensor T_all = T_all_scaled*T_std + T_mean;
    torch::Tensor T_flat_cpu = T_all.view({-1}).to(torch::kCPU).contiguous();

    auto Tacc = T_flat_cpu.accessor<torch_real,1>();
    forAll(vf_nn, cellI) vf_nn[cellI] = static_cast<scalar>(Tacc[cellI]); // OpenFOAM scalar likely double
    vf_nn.correctBoundaryConditions();

    error_c = Foam::mag(vf - vf_nn);

    const scalarField diff = mag(vf.internalField() - vf_nn.internalField());
    Info<< "max(|T - T_nn|)  = " << gMax(diff) << nl
        << "mean(|T - T_nn|) = " << gAverage(diff) << nl << endl;

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

