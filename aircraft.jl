
##############################################################################
# An aircraft is controlled by state feedback. This is an example of a linear
# quadratic regulator.
# ----------------------------------------------------------------------------
# This material was developed as part of the course EE103 taught at Stanford
# University by Professor Stephen Boyd. For details, see the course
# website at <web.stanford.edu/class/ee103/>.
# ----------------------------------------------------------------------------
# This material was modified by Jongho Kim for implementation of 
# Policy Gradient with LQR
##############################################################################

using LinearLeastSquares;
using PyPlot

# original data as taken from bryson (and ee263 slides)
A = [ -.003  .039     0 -.322;
      -.065 -.319  7.74     0;
       .020 -.101 -.429     0;
          0     0     1     0 ];
B = [   .01    1;
       -.18 -.04;
      -1.16 .598;
          0    0 ];
n = size(B,1); m = size(B,2);

# discretize
dt = 1;
B = A\(expm(A*dt) - eye(n))*B;
A = expm(A*dt);

# lqr params
T = 100;
rho = 100;

# form least norm problem with variable
# x = (x_1, sqrt(rho)*u_1, ..., sqrt(rho)*u_{T-1}, x_T)

C = [eye(n,n) zeros(n, (n+m)*(T-1))];
for t = 1:T-1
  C = [C; zeros(n, (n+m)*(t-1)) A 1/sqrt(rho)*B -eye(n,n) zeros(n, (n+m)*(T-t-1))];
end
M = C'/(C*C');
K = M[n+1:n+m, 1:n]/sqrt(rho)

# simulate
T_sim = 100;
x_cl = zeros(n, T_sim);
x_cl[:,1] = [0; 0; 0; 1];
x_ol = zeros(n, T_sim);
x_ol[:,1] = [0; 0; 0; 1];
for t = 1:T_sim-1
  x_ol[:,t+1] = A*x_ol[:,t];
  x_cl[:,t+1] = (A + B*K)*x_cl[:,t];
end

# Plot the results of open/closed loop control
x = 0:1:T_sim-1
y = zeros(x)
plot(x, y)
xlabel("t")
ylabel("u")
title("openloop input")

x = 0:1:T_sim-1
for i=1:n
    y = x_ol[i,:]
    plot(x,y)
end
xlabel("t")
ylabel("x")
title("openloop state")

x = 0:1:T_sim-1
for i=1:m
    y = (K*x_cl)[i,:]
    plot(x,y)
end
xlabel("t")
ylabel("u")
title("closedloop input")

x = 0:1:T_sim-1
for i=1:n
    y = x_cl[i,:]
    plot(x,y)
end
xlabel("t")
ylabel("x")
title("closedloop state")

# Compute time-varying LQR gains
K_vec = zeros(100-2+1,n*m)
for T = 2:100
  C = [eye(n) zeros(n, (n+m)*(T-1))];
  for t = 1:T-1
    C = [C; zeros(n, (n+m)*(t-1)) A 1/sqrt(rho)*B -eye(n) zeros(n, (n+m)*(T-t-1))];
  end
  M = C'/(C*C');
  K = M[n+1:n+m, 1:n]/sqrt(rho);
  K_vec[t,:] = K[:];
end

# Plot time-varying LQR gains
x = 0:1:T_sim-2
for i=1:n*m
    y = K_vec[:,i]
    plot(x,y)
end
xlabel("T")
ylabel("K")
title("gains vs horiz")
savefig("gains_vs_horiz")

K_star = reshape(-K_vec[99,:], (2,4))

Q = eye(n)
R = rho*eye(m)


# Policy gradient of LQR with gradient oracle
K0 = zeros(m,n)
K = K0
x0 = x_cl[:,1]
xt = x0
eta = (1/norm(A, 2)).^2
sigmaK = zeros(n,n)
ths = 1e-6

cost = 0
for i=1:T-1
    xt = x_cl[:,i]
    ut = K_star*xt
    cost = cost + xt'*Q*xt + ut'*R*ut
end
C_star = cost + norm(x_cl[:,end],2)

Pk = Q

for t=1:100000
    #xt = x_cl[:,t]
    sigmaK = sigmaK + xt*xt'
    
    C_K = x0'*Pk*x0
    # check convergence
    if C_star - C_K < ths
        println(t)
        break;
    end
    Pk = Q + K'*R*K + (A-B*K)'*Pk*(A-B*K)
    # calculate gradient
    grad_CK = 2*((R + B'*Pk*B)*K - B'*Pk*A)*sigmaK
    
    # update
    K = K - eta*grad_CK
    
    xt = (A - B*K)*xt
end
K
