using LinearAlgebra
#using PyPlot
#using Printf
include("parisi_sk.jl");

function droplower(A::SparseMatrixCSC)
    m,n = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    V = Vector{eltype(A)}()
    I = Vector{Int}()
    J = Vector{Int}()
    for i=1:n
        for j in nzrange(A,i)
            rows[j]>i && break
            push!(I,rows[j])
            push!(J,i)
            push!(V,vals[j])
        end
    end
    return sparse(I,J,V,m,n)
end

# returns phi(t,x), dphi/dx(t,x) and d^2phi/dx^2(t,x) #
function parisi_pde_data( gamma, xmax, nx )
     mtol =  0.00001;
     #nx= 500;
     #xmax=10;
     (qv, ms) = gamma;
     nq = length(qv);
     dq = 1. /nq;
     dx = 1. /nx;
     nx0 = Int(floor(1.2*nx));
     x = convert(Array{Float64},LinRange(-xmax,xmax,2*nx+1));
     x0max = Int(floor(nx0*(xmax/nx)));
     x0 = convert(Array{Float64},LinRange(-x0max,x0max,2*nx0+1));

     G = heat_kernel(x,x0,dq);

     phi = zeros(nq+1,2*nx+1);
     u = zeros(nq+1,2*nx+1);     # dphi/dx
     w = zeros(nq+1,2*nx+1);     # d^2phi/dx^2
     # terminal condition
     phi[nq+1,:] = abs.(x);
     u[nq+1,:] = sign.(x);
     w[nq+1,nx+1] = 2. /dx;
     #

     #phi0 = zeros(2*nx0+1)
     u0 = zeros(2*nx0+1);
     u0[1:nx0-nx] = -ones(nx0-nx);
     u0[nx0+nx+2:2*nx0+1] = ones(nx0-nx);
     w0 = zeros(2*nx0+1);
     #z0 = ones(2*nx0+1);
     #z1 = ones(2*nx0+1);
     toadd =  zeros(2*nx0+1);
     toadd[1:nx0-nx] = abs.(x0[1:nx0-nx] .- x0[nx0-nx+1]);
     toadd[nx0+nx+2:2*nx0+1] = abs.(x0[nx0+nx+2:2*nx0+1] .- x0[nx0+nx+1]);

     for iq in nq:-1:1
         m = ms[iq];

         phi0 = zeros(2*nx0+1);
         phi0[nx0-nx+1:nx0+nx+1] = phi[iq+1,:];
         boundary_add = zeros(2*nx0+1);
         boundary_add[1:nx0-nx] = phi[iq+1,1]*ones(nx0-nx);
         boundary_add[nx0+nx+2:2*nx0+1] = phi[iq+1,2*nx+1]*ones(nx0-nx);
         phi0 = phi0+toadd+boundary_add;
         u0[nx0-nx+1:nx0+nx+1] = u[iq+1,:];
         w0[nx0-nx+1:nx0+nx+1] = w[iq+1,:];
         z0 = exp.(m*phi0);
         z1 = G*z0;

         if m>mtol
             phi[iq,:] = (1/m)*log.(z1);
             u[iq,:] = (G*(u0.*z0))./z1;
             w[iq,:] = (G*(w0.*z0))./z1 + m*(G*(u0.*u0.*z0))./z1 - m*u[iq,:].*u[iq,:];
         else
             phi[iq,:] = G*phi0;
             u[iq,:] = G*u0;
             w[iq,:] = G*w0;
         end
     end
     phi, u, w
end

function uf( iq, xm, dph, xmax, nx, l, r )
    #u = zeros(nx);
    dx = xmax/nx;
    pl = r*(xm .>= xmax);
    mi = l*(xm .<= -xmax);
    cent = convert(Array{Int64,1}, (xm .> -xmax) .& (xm .< xmax));
    i1 = convert(Array{Int64,1}, floor.(xm/dx)) .+ (nx+1);
    i1 = min.(i1,2*nx-1);
    i1 = max.(i1,1);
    s = convert(Array{Int64,1}, sign.(xm));
    zs = convert(Array{Int64,1}, (s .== 0) );
    s = s+zs;
    i2 =  i1+s;
    i2 = min.(i2,2*nx);
    i2 = max.(i2,0);
    u = (-(xm .- (i2 .- nx)*dx).*dph[iq,i1 .+ 1] + (xm - (i1 .- nx)*dx).*dph[iq,i2 .+ 1])./(dx*s);
    u = cent.*u+pl+mi;
    u
end


# iamp
function amp( A, gamma, dph, ddph, xmax, nx )
    tol = 0.000000001;
    n = size(A)[1];
    (qv, gammav) = gamma;
    nq = length(qv);
    dq = qv[2]-qv[1];
    sdq = sqrt(dq);
    sn = sqrt(n);

    z = zeros(n,nq+2);
    x = zeros(n,nq+2);
    u = zeros(n,nq+2);
    m = zeros(n,nq+2);
    ons = zeros(n,nq+2);
    b = zeros(nq+2);
    delz = zeros(n,nq+1);
    #u[:,1] = zeros(n);
    u[:,2] = ones(n);
    #m[:,1] = zeros(n);
    z[:,2] = sdq*randn(n);
    delz[:,1] = z[:,2];#sdq*ones(n);
    m[:,2] =sdq*randn(n);# z[:,2];#
    #ons[:,2]=m[:,2];
    b[2] = 1;
    I = zeros(n);
    energy = zeros(nq);
    for iq = 2:nq+1

        #sigma = sign.(m[:,iq]);
        energy[iq-1] = dot(m[:,iq],A*m[:,iq])/(2*n);

        ### AMP iteration
        z[:,iq+1] = A*m[:,iq] - ons[:,iq-1];

        ### orthogonalize
        delz[:,iq] = z[:,iq+1] - z[:,iq];
        delz[:,iq] = project(delz[:,iq],delz,sn,sdq,max(1,iq-5),iq-1);#
        delz[:,iq] = sdq*sn*delz[:,iq]/norm(delz[:,iq]); # not necessary
        z[:,iq+1] = z[:,iq] + delz[:,iq];

        ####
        #m[:,iq+1] = m[:,iq] + u[:,iq].*(z[:,iq+1]-z[:,iq]);

        ###
        I = I + (b[iq] - b[iq-1])*m[:,iq-1];
        ons[:,iq] =  b[iq]*m[:,iq] - I;
        #ons[:,iq] =  b[iq]*(m[:,iq]-m[:,iq-1]) + b[iq-1]*m[:,iq-1];
        #ons[:,iq] =  b[iq]*m[:,iq];

        x[:,iq+1] = x[:,iq] + gammav[iq-1]*uf(iq-1,x[:,iq],dph,xmax,nx,-1,1)*dq + (z[:,iq+1]-z[:,iq]);
        m[:,iq+1] = uf(iq,x[:,iq+1],dph,xmax,nx,-1,1);
        m[:,iq+1] = sqrt(iq*dq)*sn*m[:,iq+1]/norm(m[:,iq+1]); # not necessary
        ###
        v = uf(iq,x[:,iq+1],ddph,xmax,nx,0,0);
        u[:,iq+1] = sn*v/(norm(v)+tol);
        b[iq+1] = sum(u[:,iq+1])/n;
    end
    m, x, z, energy
end

function project(del,z,sn,sdq,iq1,iq2)
        #k = length(z,2);
        del1 = zeros(length(del));
        for ik=iq2:-1:iq1
            del1 = del1 + dot(del,z[:,ik])*z[:,ik]/dot(z[:,ik],z[:,ik]);
        end
        del =  del - del1;
        del = sn*sdq*del/norm(del);
        del
end

function energy_AMP_stats(nn_array,nb_samples)
    nn = length(nn_array);
    #nb_samples = 10;
    energy_vals = zeros(nb_samples,nn,4);
    energy_means = zeros(nn,4);
    energy_std = zeros(nn,4);
    #nn_array=[2000,4000];
    #nn_array=[500,750,1000,1250];
    ii=1;#jj=1;
    for N in nn_array
      nq_array = convert(Array{Int32},floor.([N/4,N/3,N/2,N]));
      jj=1;
      for nq in nq_array
        #nq = 750;
        qv = float(collect(0:nq-1))/nq;
        spl = Spline1D(gamma[1], gamma[2]; k=5); # fits spline of degree 5
        gammav = spl(qv); # evaluates spline at qv
        gammanew = [qv,gammav];
        phi, dphi, ddphi= parisi_pde_data(gammanew,xmax,nx);

        for iter=1:nb_samples
            #A = sprandn(N,N,dens);
            #A = droplower(A);
            #A = Symmetric(A)/sqrt(dens*N);
            A = randn(N,N);
            A = (A+A')/sqrt(2*N);
            m,x,z,energy = amp(A,gammanew,dphi,ddphi,xmax,nx);
            #sigma = sign.(m[:,nq+1]);
            #energy_vals[iter,ii,jj] = dot(sigma,A*sigma)/(2*N);#
            energy_vals[iter,ii,jj] = energy[end];
        end
        energy_means[ii,jj] = mean(energy_vals[:,ii,jj]);
        energy_std[ii,jj] = std(energy_vals[:,ii,jj]);
        jj=jj+1;
      end
      ii=ii+1;
    end
    energy_vals, energy_means, energy_std
end


function derv( f, xmax, nx )
    dx = xmax/nx;
    df = zeros(2*nx+1);
    ix = collect(1:2*nx-1);
    df[2:2*nx] = f[ix .+ 2] - f[ix];
    df/(2*dx)
end



function px( gamma, dph, xmax, nx )
    x = convert(Array{Float64},LinRange(-xmax,xmax,2*nx+1));
    dx = x[2]-x[1];
    qv, gammav = gamma;
    nq = length(qv);
    prx = zeros(nq,2*nx+1);
    prx[1,nx+1] = 1/dx;
    #prx[1,nx] = 1/dx;
    dq = qv[2]-qv[1];

    G = heat_kernel(x,x,dq);
    for iq = 1:nq-1
        v = gammav[iq]*uf(iq,x,dph,xmax,nx,-1,1);
        prx[iq+1,:] = G*prx[iq,:];
        prx[iq+1,:] = prx[iq+1,:]-derv(v.*prx[iq+1,:],xmax,nx)*dq;
        prx[iq+1,:] = prx[iq+1,:]/(sum(prx[iq+1,:])*dx);
    end
    prx
end


# N , nq , nx, xmax,

function save_energy_vals(filename , energy_vals )
    nq = length(energy_vals);
    io = open(filename, "w");
    println(io,"#          \n");
    println(io,"#   E_N       \n");
    for i=1:nq
        @printf(io, "%.10f \n", energy_vals[i]);
    end
    close(io);
end
