using PyPlot
using SparseArrays
using Statistics
using LinearAlgebra
using Dierckx
#include("parisi_sk.jl");
include("iamp_sk.jl");

# Runs IAMP on a random Gaussian matrix

# Load functional order parameter (fop) at zero temperature (precomputed)
gamma = read_file("fop_sk/sk_250rsb_500it_7631685.txt");
nq = 750; # nb of RSB
qv = float(collect(0:nq-1))/nq; # discretization of [0,1]
spl = Spline1D(gamma[1], gamma[2]; k=5); # fits spline of degree 5
gammav = spl(qv); # evaluates spline at qv
gammanew = [qv,gammav]; # new order parameter

# Solve the Parisi PDE
xmax=10;
nx=500;
phi, dphi, ddphi= parisi_pde_data(gammanew,xmax,nx);

# Genrate the disorder matrix
N=2000;
A = randn(N,N);
A = (A+A')/sqrt(2*N);
#dens = .2;
#A = sprandn(N,N,dens);
#A = droplower(A);
#A = Symmetric(A)/sqrt(dens*N);

# Run the algorithm
m,x,z,energy = amp(A,gammanew,dphi,ddphi,xmax,nx); # ouputs an entire trajectory along a branch of the ultrametric tree
# m : size N x (nq+1). Column i is the barycenter of the ancestor state at level i/nq.
# x : size N x (nq+1). Column i is the vector of cavity fields at level i/nq.
# z: size N x (nq+1). the AMP iterate (see the function amp)
#energy: the energy trajectory (should in principle be increasing with time)

sigma = sign.(m[:,nq+1]); # rounded solution (taken at a leaf of the tree)
E_N = dot(sigma,A*sigma)/(2*N); # its energy
plot(gammanew[1],energy,color="black",linewidth = 2); # plot energy trajectory
scatter([1],[E_N],color="red");
xlabel(L"$t$");
ylabel(L"$E_N$");
title(L"$SK ~:~ E_N = N^{-1}H_N(m^t)$");








########## Plots of fop gamma for SK #####
gamma_50 = read_file("fop_sk/sk_50rsb_500it_763176.txt");
gamma_100 = read_file("fop_sk/sk_100rsb_500it_7631699.txt");
gamma_150 = read_file("fop_sk/sk_150rsb_500it_763168.txt");
gamma_200 = read_file("fop_sk/sk_200rsb_500it_763169.txt");
plot(gamma_50[1],gamma_50[2]);
plot(gamma_100[1],gamma_100[2]);
plot(gamma_150[1],gamma_150[2]);
plot(gamma_200[1],gamma_200[2]);
xlabel(L"$t$");
ylabel(L"\gamma_\star(t)");
title("2-spin");
legend(["50RSB","100RSB","150RSB","200RSB"]);
savefig("data/fop_sk/gamma_2spin_plots.pdf");
