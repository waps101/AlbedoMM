function AO = MR_ambient_occlusion(V,F,N,subdiv)
%MR_AMBIENT_OCCLUSION Per-vertex ambient occlusion for a mesh
%   Inputs:
%     V      - nverts x 3 matrix of vertex positions
%     F      - nfaces x 3 matrix of face indices
%     N      - nverts x 3 matrix of per-vertex surface normals
%     subdiv - number of subdivisions of icosahedron to use for view
%              sampling sphere. 1 is very coarse, 2 is better, 3 is very 
%              accurate (42, 162, 642 samples respectively)
%
%   Outputs:
%     AO     - nverts x 3 per-vertex ambient occlusion values (0..1)
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


[dirs,~]=icosphere(subdiv);

cameraparams.type = 'scaledorthographic';
cameraparams.scale = 1;

AO = zeros(size(V,1),1);
count = zeros(size(AO));
for j=1:size(dirs,1)
    weights = dirs(j,1).*N(:,1) + dirs(j,2).*N(:,2) + dirs(j,3).*N(:,3);
    weights(weights<0)=0;
    axis = cross(dirs(j,:),[0;0;-1]);
    if norm(axis)==0
        R = eye(3);
    else
        axis = axis./norm(axis);
        angle = acos(dot(dirs(j,:),[0;0;-1]));
        R = axang2rotm(axis.*angle);
    end
    t = [0;0;0];
    cameraparams.T = [R t];
    
    [Vxy,Vcam] = MR_project(V,cameraparams);
    scale = 500 / max( max(Vxy(:,1))-min(Vxy(:,1)), max(Vxy(:,2))-min(Vxy(:,2)) );
    Vxy = Vxy.*scale;
    Vxy(:,1) = Vxy(:,1) - min(Vxy(:,1));
    Vxy(:,2) = Vxy(:,2) - min(Vxy(:,2));
    
    cameraparams.w = ceil(max(Vxy(:,1)));
    cameraparams.h = ceil(max(Vxy(:,2)));
    
    [zbuffer,fbuffer,wbuffer,Vz] = MR_rasterise_mesh_mex(uint32(F), Vcam, Vxy, cameraparams.w, cameraparams.h);
    visibility = MR_vertex_visibility(Vxy,Vz,zbuffer,fbuffer,F);
    AO = AO + visibility.*weights;

    count = count+(weights>0);
    %disp(['Direction ' num2str(j) ' of ' num2str(size(dirs,1))]);
end
AO = AO./count;
AO = min(1,2.*AO);

end

function [vv,ff] = icosphere(varargin)
%ICOSPHERE Generate icosphere.
% Create a unit geodesic sphere created by subdividing a regular
% icosahedron with normalised vertices.
%
%   [V,F] = ICOSPHERE(N) generates to matrices containing vertex and face
%   data so that patch('Faces',F,'Vertices',V) produces a unit icosphere
%   with N subdivisions.
%
%   FV = ICOSPHERE(N) generates an FV structure for using with patch.
%
%   ICOSPHERE(N) and just ICOSPHERE display the icosphere as a patch on the
%   current axes and does not return anything.
%
%   ICOSPHERE uses N = 3.
%
%   ICOSPHERE(AX,...) plots into AX instead of GCA.
%
%   See also SPHERE.
%
%   Based on C# code by Andres Kahler
%   http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
%
%   Wil O.C. Ward 19/03/2015
%   University of Nottingham, UK

% Parse possible axes input
if nargin > 2
    error('Too many input variables, must be 0, 1 or 2.');
end
[cax,args,nargs] = axescheck(varargin{:});

n = 3; % default number of sub-divisions
if nargs > 0, n = args{1}; end % override based on input

% generate regular unit icosahedron (20 faced polyhedron)
[v,f] = icosahedron(); % size(v) = [12,3]; size(f) = [20,3];

% recursively subdivide triangle faces
for gen = 1:n
    f_ = zeros(size(f,1)*4,3);
    for i = 1:size(f,1) % for each triangle
        tri = f(i,:);
        % calculate mid points (add new points to v)
        [a,v] = getMidPoint(tri(1),tri(2),v);
        [b,v] = getMidPoint(tri(2),tri(3),v);
        [c,v] = getMidPoint(tri(3),tri(1),v);
        % generate new subdivision triangles
        nfc = [tri(1),a,c;
            tri(2),b,a;
            tri(3),c,b;
            a,b,c];
        % replace triangle with subdivision
        idx = 4*(i-1)+1:4*i;
        f_(idx,:) = nfc;
    end
    f = f_; % update
end

% remove duplicate vertices
[v,b,ix] = unique(v,'rows'); clear b % b dummy / compatibility
% reassign faces to trimmed vertex list and remove any duplicate faces
f = unique(ix(f),'rows');

switch(nargout)
    case 0 % no output
        cax = newplot(cax); % draw to given axis (or gca)
        showSphere(cax,f,v);
    case 1 % return fv structure for patch
        vv = struct('Vertices',v,'Faces',f,...
            'VertexNormals',v,'FaceVertexCData',v(:,3));
    case 2 % return vertices and faces
        vv = v; ff = f;
    otherwise
        error('Too many output variables, must be 0, 1 or 2.');
end

end

function [i,v] = getMidPoint(t1,t2,v)
%GETMIDPOINT calculates point between two vertices
%   Calculate new vertex in sub-division and normalise to unit length
%   then find or add it to v and return index
%
%   Wil O.C. Ward 19/03/2015
%   University of Nottingham, UK

% get vertice positions
p1 = v(t1,:); p2 = v(t2,:);
% calculate mid point (on unit sphere)
pm = (p1 + p2) ./ 2;
pm = pm./norm(pm);
% add to vertices list, return index
i = size(v,1) + 1;
v = [v;pm];

end

function [v,f] = icosahedron()
%ICOSAHEDRON creates unit regular icosahedron
%   Returns 12 vertex and 20 face values.
%
%   Wil O.C. Ward 19/03/2015
%   University of Nottingham, UK
t = (1+sqrt(5)) / 2;
% create vertices
v = [-1, t, 0; % v1
    1, t, 0; % v2
    -1,-t, 0; % v3
    1,-t, 0; % v4
    0,-1, t; % v5
    0, 1, t; % v6
    0,-1,-t; % v7
    0, 1,-t; % v8
    t, 0,-1; % v9
    t, 0, 1; % v10
    -t, 0,-1; % v11
    -t, 0, 1];% v12
% normalise vertices to unit size
v = v./norm(v(1,:));

% create faces
f = [ 1,12, 6; % f1
    1, 6, 2; % f2
    1, 2, 8; % f3
    1, 8,11; % f4
    1,11,12; % f5
    2, 6,10; % f6
    6,12, 5; % f7
    12,11, 3; % f8
    11, 8, 7; % f9
    8, 2, 9; % f10
    4,10, 5; % f11
    4, 5, 3; % f12
    4, 3, 7; % f13
    4, 7, 9; % f14
    4, 9,10; % f15
    5,10, 6; % f16
    3, 5,12; % f17
    7, 3,11; % f18
    9, 7, 8; % f19
    10, 9, 2];% f20
end