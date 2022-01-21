function grav = GravityForces(thetalist, g, Mlist, Glist, Slist)
% *** CHAPTER 8: DYNAMICS OF OPEN CHAINS ***
% Takes thetalist: A list of joint variables,
%       g: 3-vector for gravitational acceleration,
%       Mlist: List of link frames i relative to i-1 at the home position,
%       Glist: Spatial inertia matrices Gi of the links,
%       Slist: Screw axes Si of the joints in a space frame, in the format
%              of a matrix with the screw axes as the columns.
% Returns grav: The joint forces/torques required to overcome gravity at 
%               thetalist
% This function calls InverseDynamics with Ftip = 0, dthetalist = 0, and 
% ddthetalist = 0.
% Example Input (3 Link Robot):
% 
% clear; clc;
% thetalist = [0.1; 0.1; 0.1];
% g = [0; 0; -9.8];
% M01 = [[1, 0, 0, 0]; [0, 1, 0, 0]; [0, 0, 1, 0.089159]; [0, 0, 0, 1]];
% M12 = [[0, 0, 1, 0.28]; [0, 1, 0, 0.13585]; [-1, 0 , 0, 0]; [0, 0, 0, 1]];
% M23 = [[1, 0, 0, 0]; [0, 1, 0, -0.1197]; [0, 0, 1, 0.395]; [0, 0, 0, 1]];
% M34 = [[1, 0, 0, 0]; [0, 1, 0, 0]; [0, 0, 1, 0.14225]; [0, 0, 0, 1]];
% G1 = diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7]);
% G2 = diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393]);
% G3 = diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275]);
% Glist = cat(3, G1, G2, G3);
% Mlist = cat(3, M01, M12, M23, M34); 
% Slist = [[1; 0; 1;      0; 1;     0], ...
%        [0; 1; 0; -0.089; 0;     0], ...
%        [0; 1; 0; -0.089; 0; 0.425]];
% grav = GravityForces(thetalist, g, Mlist, Glist, Slist)
% 
% Output:
% grav =
%   28.4033
%  -37.6409
%   -5.4416

n = size(thetalist, 1);
grav = InverseDynamics(thetalist, zeros(n, 1), zeros(n, 1) , g, ...
                       [0; 0; 0; 0; 0; 0], Mlist, Glist, Slist);
end