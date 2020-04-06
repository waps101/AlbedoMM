function pervertexcolour = MR_poisson_blending(V,F,samples,lambda)
%MR_POISSON_BLENDING Screened Poisson blending on a mesh
%   Inputs:
%     V       - nverts x 3 matrix of vertex positions
%     F       - nfaces x 3 matrix of face indices
%     samples - nviews x 1 structure with:
%        samples(i).pervertexcolour - nverts x 3 matrix of sampled colours
%        samples(i).weights - nverts x 3 matrix of blending weight
%        Note: samples(1).pervertexcolour acts as the screening term - in
%        other words the overall unknown colour offset is solved to
%        minimise error to these colours (ignoring vertices with zero
%        weight). This is used instead of a boundary condition.
%     lambda  - regularisation weight
%
%   Outputs:
%     pervertexcolour - nverts x 3 per-vertex blended colours
%
% Triangles where at least one vertex has no valid colour samples are
% assigned zero gradient. The effect is to smoothly inpaint these regions
% based on the surrounding colours.
%
% An extension to the Matlab Renderer
% (https://github.com/waps101/MatlabRenderer)
% 
% This code was written for the following paper which you should cite if
% you use the code in your research:
%
% William A. P. Smith, Alassane Seck, Hannah Dee, Bernard Tiddeman, Joshua
% Tenenbaum and Bernhard Egger. A Morphable Face Albedo Model. In Proc.
% CVPR, 2020.
%
% William Smith
% University of York
% 2020

X = V';
F = F';

n = size(X,2);
m = size(F,2);

% Callback to get the coordinates of all the vertex of index i=1,2,3 in all faces
XF = @(i)X(:,F(i,:));

% Compute un-normalized normal through the formula e1xe2 where ei are the edges.
Na = cross( XF(2)-XF(1), XF(3)-XF(1) );

% Compute the area of each face as half the norm of the cross product.
amplitude = @(X)sqrt( sum( X.^2 ) );
A = amplitude(Na)/2;

% Compute the set of unit-norm normals to each face.
normalize = @(X)X ./ repmat(amplitude(X), [3 1]);
N = normalize(Na);

% Populate the sparse entries of the matrices for the operator implementing ?i?fui(Nf?ei)
I = []; J = []; V = []; % indexes to build the sparse matrices
for i=1:3
    % opposite edge e_i indexes
    s = mod(i,3)+1;
    t = mod(i+1,3)+1;
    % vector N_f^e_i
    wi = cross(XF(t)-XF(s),N);
    % update the index listing
    I = [I, 1:m];
    J = [J, F(i,:)];
    V = [V, wi];
end

% Sparse matrix with entries 1/(2Af)
dA = spdiags(1./(2*A(:)),0,m,m);

% Compute gradient.
GradMat = {};
for k=1:3
    GradMat{k} = dA*sparse(I,J,V(k,:),m,n);
end

V = X';
F = F';


y = zeros(size(F,1)*3,3);

Tweight = zeros(size(F));
for im=1:length(samples)
    Tweight(:,im) = min([samples(im).weight(F(:,1)) samples(im).weight(F(:,2)) samples(im).weight(F(:,3))],[],2);
    rhs{im} = [GradMat{1}; GradMat{2}; GradMat{3}]*samples(im).pervertexcolour;
end

[~,idx] = max(Tweight,[],2);

for im=1:length(samples)
    mask = (idx==im) & (Tweight(:,im)>0);
    mask = [mask; mask; mask];
    y(mask,:) = rhs{im}(mask,:);
end

A = [GradMat{1}; GradMat{2}; GradMat{3}; sparse(1:size(V,1),1:size(V,1),lambda.*(samples(1).weight>0),size(V,1),size(V,1))];

y = [y; lambda.*samples(1).pervertexcolour];

pervertexcolour = A\y;

end

