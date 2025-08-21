# IAMP Algoritm for optimizing the Sherrington-Kirkpatrick Hamiltonian

This is an implementation of Incremental Approximate Message Passing for solving the random optimization problem 

$$
{\large \max   \langle  {x}, {A} {x}\rangle \\;\\;\\;\\;\text{  subj. to    } {x}\in \\{+1 ,-1\\}^n}
$$

where $A$ is a Wigner matrix, i.e. a symmetriv matrix with $A_{ij}$, $i<j$ independent, sub-Gaussian, with 0-mean and 
variance $1/n$, and $A_{ii}=O(n^{-1/2})$.

Files:
- small_script_sk.jl  executes the code. 
- iamp_sk.jl contains an implementation of IAMP. amp is the main routine. 
- parisi_sk.jl contains code for computing the optimal functional order parameter (fop) gamma at zero temperature. proj_grad is the main routine. 
- fop_sk contains precomputed fop's at different levels of RSB, i.e, different levels of discretizations of the interval [0,1].
- The file naming convention is:  sk_levels of RSB_nb of iterations of the solver  proj_grad_ energy achieved

[Note that the algorithm performace depends on the accuracy of the fop]

The code was developed and used for numerical experiments in:

- Alaoui, A.E. and Montanari, A., 2020. Algorithmic thresholds in mean field spin glasses. arXiv:2009.11481

Please, refer to the paper above if you use our code.

Proofs:
- Montanari, A., 2021. Optimization of the Sherrington--Kirkpatrick Hamiltonian. SIAM Journal on Computing, (0), pp.FOCS19-1.
- El Alaoui, A., Montanari, A. and Sellke, M., 2021. Optimization of mean-field spin glasses. The Annals of Probability, 49(6), pp.2922-2960.
  
