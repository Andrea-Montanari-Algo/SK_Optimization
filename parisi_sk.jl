using LinearAlgebra
#using PyPlot
using Printf
### Routines computing the solution to the Parisi PDE at zero temperature and its gradient with respect to the oder parameter gamma ###
### Order parameter is the cumultive distribution function of the Parisi measure. It is a non-negative step-wise function on [0,1]. ###

# Intialize oder parameter #

function init_gamma( nq )
    q_val = float(collect(0:nq-1))/nq; #cumsum(ones(nq),1)/nq;
    gamma_val = float(collect(0:nq-1))/nq;#ones(nq,1)/nq;
    q_val,gamma_val
end


## extrapolate gamma to new grid ##

function extrapolate(gamma,nqnew)
    qv = gamma[1];
    gammav = gamma[2];
    nq = length(qv);
    qvnew = float(collect(0:nqnew-1))/nqnew;
    #coeffs = Vandermonde(qv)\gammav;
    gammavnew = zeros(nqnew);
    #gammavnew[1] = gammav[1];
    for iq=1:nqnew
        #gammavnew[iq] = dot(coeffs,powers(qvnew[iq],nq));
        jq = Int(floor(nq*(iq-1)/nqnew))+1;
        gammavnew[iq] = gammav[jq];
    end
    qvnew,gammavnew
end


# Heat kernel for time increment dt #
function heat_kernel( x, x0 , dt)
    nnx = length(x);
    nnx0 = length(x0);
    G = zeros(nnx,nnx0);
    for i=1:nnx
        for j=1:nnx0
            G[i,j] = exp(-(x[i]-x0[j])*(x[i]-x0[j])/(2*dt));
        end
    end
    s=G*ones(nnx0);
    for i=1:nnx
        G[i,:] = G[i,:]/s[i];
    end
    G
end

# Solution to the Parisi PDE Phi(0,x) with input mu and terminal condition Phi(1,x) = |x|#
function parisi_pde( gamma , xmax , nq , nx , nx0 , G , xabs , toadd )
    mtol =  0.00001;
    phi = ones(nq+1,2*nx+1);
    # terminal condition
    phi[nq+1,:] = xabs;
    #beta=2;
    #phi[nq+1,:] = (1/beta)*log.(cosh.(beta*x));
    #
    (qv, ms) = gamma;
    #phi0 = zeros(2*nx0+1);
    #u0 = zeros(2*nx0+1);
    #z0 = ones(2*nx0+1);
    #z1 = ones(2*nx0+1);

    for iq in nq:-1:1 #range(nq-1,-1,-1):
        m = ms[iq];

        phi0 = zeros(2*nx0+1);
        phi0[nx0-nx+1:nx0+nx+1] = phi[iq+1,:];
        boundary_add = zeros(2*nx0+1);
        boundary_add[1:nx0-nx] = phi[iq+1,1]*ones(nx0-nx);
        boundary_add[nx0+nx+2:2*nx0+1] = phi[iq+1,2*nx+1]*ones(nx0-nx);
        phi0 = phi0+toadd+boundary_add;

        if m>mtol
            phi[iq,:] = (1/m)*log.(G*exp.(m*phi0));
        else
            phi[iq,:] = G*phi0;
        end
    end
    phi
end

# value at (0,0) and its gradient with respect to gamma#
function phi_gradphi( gamma , xmax , nq , nx , nx0 , G , xabs , toadd )
     mtol =  0.00001;
     phi = ones(nq+1,2*nx+1);
     u = zeros(nq,nq,2*nx+1);
     # terminal condition
     phi[nq+1,:] = xabs;
     #beta=2;
     #phi[nq+1,:] = (1/beta)*log.(cosh.(beta*x));
     #
     (qv, ms) = gamma;
     #phi0 = zeros(2*nx0+1);
     #u0 = zeros(2*nx0+1);
     #z0 = ones(2*nx0+1);
     #z1 = ones(2*nx0+1);


     for iq in nq:-1:1
         m = ms[iq];

         phi0 = zeros(2*nx0+1);
         phi0[nx0-nx+1:nx0+nx+1] = phi[iq+1,:];
         boundary_add = zeros(2*nx0+1);
         boundary_add[1:nx0-nx] = phi[iq+1,1]*ones(nx0-nx);
         boundary_add[nx0+nx+2:2*nx0+1] = phi[iq+1,2*nx+1]*ones(nx0-nx);
         phi0 = phi0+toadd+boundary_add;

         z0 = exp.(m*phi0);
         z1 = G*z0;

         if m>mtol
             phi[iq,:] = (1/m)*log.(z1);
             u[iq,iq,:] = (1/m)*( (G*(phi0.*z0))./z1 - phi[iq,:]);
         else
             phi[iq,:] = G*phi0;
             u[iq,iq,:] = 0.5*(G*(phi0.*phi0)-phi[iq,:].*phi[iq,:]);
         end

         if iq < nq
             for jq=iq+1:nq
                 #u0 = zeros(2*nx0+1);
                 #u0[nx0-nx+1:nx0+nx+1] = u[iq+1,jq,:];
                 uu = u[iq+1,jq,:].*z0[nx0-nx+1:nx0+nx+1];
                 #lv= uu[1]; uv = uu[2*nx+1];
                 #u0 = [lv*ones(nx0-nx);uu;uv*ones(nx0-nx)];
                 u0 = [zeros(nx0-nx);uu;zeros(nx0-nx)];
                 u[iq,jq,:] = (G*u0)./z1;
             end
         end
     end

     # value at (0,0)
     f = phi[1,nx+1];
     #gradient at (0,0) w.r.t. gamma = cdf
     gf = u[1,:,nx+1];

     #gradient at (0,0) w.r.t. mu = pdf
     #gf = zeros(nq);
     #gf[nq] = u[1,nq,nx+1];
     #for iq in nq-1:-1:1
     #  gf[iq] = gf[iq+1]+u[1,iq,nx+1];
     #end
     f, gf
end

# computes the Parisi functional phi(0,0) - .5 int_0^1 t gamma(t) dt at gamma
function parisi_functional( gamma , xmax , nq , nx , nx0 , G , xabs , toadd , delq )
    #FAC_XMAX = 8.0;
    #nvx = [300, 400, 500];
    #xmax = FAC_XMAX;
    #nn = length(nvx);
    (qv, gammav) = gamma;

    #nx= 500;
    #xmax=10;
    phi = parisi_pde( gamma , xmax , nq , nx , nx0 , G , xabs , toadd );
    fr = phi[1,nx+1] - 0.25*dot(gammav,delq);
    fr
    #fr =zeros(nn);
    #(qv, mv) = mu;
    #for i=1:nn
    #    phi = parisi_pde(mu,xmax,nvx[i]);
    #    fr[i] = phi[1,nvx[i]+1] - 0.25*dot(mv,1 .- qv.*qv);
    #end
    #
    # interpolation between the nn values
    #x = [ones(1,nn);(1 ./nvx)']; # 2 x nn
    #S = inv(x*x');
    #theta = S*(x*fr);
    #theta[1]
end

# computes the Parisi functional and its gradient at gamma
function grad_parisi_functional( gamma , xmax , nq , nx , nx0 , G , xabs , toadd , delq )
    #FAC_XMAX = 8.0;
    #nvx = [300, 400, 500];
    #xmax = FAC_XMAX;
    #nn = length(nvx);
    #fr =zeros(nn);
    #gfr = zeros(nq,nn);
    (qv, gammav) = gamma;
    #nq = length(qv);
    #delq = zeros(nq);
    #for i=1:nq-1
    #    delq[i] = qv[i+1]*qv[i+1]-qv[i]*qv[i];
    #end
    #delq[nq] = 1 - qv[nq]*qv[nq];

    #nx= 500;
    #xmax=10;
    phi, gphi = phi_gradphi( gamma , xmax , nq , nx , nx0 , G , xabs , toadd );
    fr = phi - 0.25*dot(gammav,delq);
    gfr = gphi - 0.25*delq;
    fr, gfr
    #for i=1:nn
    #    phi, gphi = phi_gradphi(mu,xmax,nvx[i]);
    #    fr[i] = phi - 0.25*dot(mv,1 .- qv.*qv);
    #    gfr[:,i] = gphi - 0.25*(1 .- qv.*qv);
    #end
    #
    # interpolation between the nn values
    #x = [ones(1,nn);(1 ./nvx)']; # 2 x nn
    #S = inv(x*x');
    #theta_fr = S*(x*fr);
    #theta_gfr = S*(x*gfr');
    #theta_fr[1],theta_gfr[1,:]
end









### optimization ####


function project_gamma(gammav,nq)
    gammav = max.(0,gammav);
    muv = diff(gammav);
#    muv = zeros(nq);
#    for i=2:nq
#        muv[i] = gammav[i]-gammav[i-1];
#    end
    muv = max.(muv,0);
    gammav[2:nq] = cumsum(muv);
    #gammanew = zeros(nq);
    #gammanew[1] = gamma[1];
#    for i=2:nq
#        gammav[i] = gammav[i-1]+muvnew[i];
#    end
    gammav
end

function backtrack_linesearch( gamma , xmax , nq , nx , nx0 , G , xabs , toadd , delq, fr, gr, step0, incr_cond)
    (q, gammav) = gamma;
    step = step0;
    i = 0;
    gap =1.;
    gammavnew = zeros(length(gammav));
    #gmax = maximum(abs.(gr));
    #gr = gr/gmax;

    if incr_cond == 1
        while (i<30) & (gap>0.)
            vec = step*gr;
            gammavnew = gammav-vec;
            gammavnew = project_gamma(gammavnew,nq); ## Forces cumulative to be positive and increasing ##
            new_fr = parisi_functional( [q,gammavnew] , xmax , nq , nx , nx0 , G , xabs , toadd , delq);
            gap = new_fr-fr+0.5*step*dot(gr,gr);
            #print(step, new_fr);
            i = i+1;
            step = .5*step;
        end
    else
        while (i<30) & (gap>0.)
            vec = step*gr;
            gammavnew = gammav-vec;
            gammavnew = max.(0,gammavnew); ## Forces cumulative to be positive ##
            new_fr = parisi_functional( [q,gammavnew] , xmax , nq , nx , nx0 , G , xabs , toadd , delq);
            gap = new_fr-fr+0.5*step*dot(gr,gr);
            #print(step, new_fr);
            i = i+1;
            step = .5*step;
        end
    end
    gammavnew, i
end


function proj_grad(nq, maxiter, gamma0 )
#### initialization ####
    incr_cond = 0; ## if = 1 then gamma is forced to be positive and incresing. If = 0 then gamma is only forced to be positive.

    #nq=100;
    (qv, gammav) = gamma0;
    #nq = length(qv);
    nx= 500;
    xmax=10;
    x = convert(Array{Float64},LinRange(-xmax,xmax,2*nx+1));

    # cumulative of parisi measure
    ms=gammav; # cumsum(mv,1);

    # terminal condition
    xabs = abs.(x);

    dq = 1. /nq;
    #sdev = 6*sqrt(dq)
    nx0 = Int(floor(1.2*nx));
    #nx0 = nx+Int(floor(nx*(sdev/xmax)));
    x0max = Int(floor(nx0*(xmax/nx)));
    x0 = convert(Array{Float64},LinRange(-x0max,x0max,2*nx0+1));

    G = heat_kernel(x,x0,dq);

    toadd =  zeros(2*nx0+1);
    toadd[1:nx0-nx] = abs.(x0[1:nx0-nx] .- x0[nx0-nx+1]);
    toadd[nx0+nx+2:2*nx0+1] = abs.(x0[nx0+nx+2:2*nx0+1] .- x0[nx0+nx+1]);

    #nq = length(qv);
    delq = zeros(nq);
    for i=1:nq-1
        delq[i] = qv[i+1]*qv[i+1]-qv[i]*qv[i];
    end
    delq[nq] = 1 - qv[nq]*qv[nq];

#### projected gradient descent with linesearch ###
    step0 = 10000;
    gamma=gamma0;
    val = zeros(maxiter);
    gfr = zeros(nq);
    for iter=1:maxiter
        fr, gfr = grad_parisi_functional( gamma , xmax , nq , nx , nx0 , G , xabs , toadd , delq );
        val[iter] = fr;
        gammavnew,i = backtrack_linesearch( gamma , xmax , nq , nx , nx0 , G , xabs , toadd , delq, fr, gfr, step0, incr_cond);
        #gammavnew = max.(0,gammavnew); ## Forces cumulative to be positive ##

        print("iter = ", iter, "   free energy = ", fr, "   backtrack iter = ", i, "\n");
    #    println(io, "iter = ", iter, "   free energy = ", fr, "   energy = ", ener, "   backtrack iter = ", i, "\n");
        gamma = [qv,gammavnew];
    end
    gamma, val, gfr
end



## save/read data ######
#savefig( "parisi_cumulative.pdf");
function save_data(filename , gamma)
    qv = gamma[1]; gammav = gamma[2];
    nq = length(qv);
    io = open(filename, "w");
    println(io,"#      q          gamma(q)       \n");
    for i=1:nq
        @printf(io, "%.10f %s %.10f \n", qv[i] , "   ", gammav[i]);
    end
    close(io);
end

using DelimitedFiles
function read_file(filename)
    data=readdlm(filename, ' ', String, '\n', skipstart=2, comments=true, comment_char='#');
    nq = size(data)[1];
    qv = zeros(nq);
    gamma = zeros(nq);
    for i=1:nq
           qv[i]=parse(Float64,data[i,1]);
           gamma[i]=parse(Float64,data[i,6]);
    end
    [qv,gamma]
end
