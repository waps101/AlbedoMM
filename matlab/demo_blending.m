% Sample script showing sampling and Poisson blending for 3 views of a face

% Sampling and blending parameters
boundarydistT = 100;
lambda = 0.1;

% Load mesh
load('sample_data/FV.mat');

% Set up intrinsic camera parameters
cameraparams.h = 3872;
cameraparams.w = 2592;
cameraparams.f = 9.1189e+03;
cameraparams.cx = 1.3237e+03;
cameraparams.cy = 2.4020e+03;
cameraparams.k1 = -0.4317;
cameraparams.k2 = 20.9181;
cameraparams.k3 = -413.8037;
cameraparams.p1 = 0.0012;
cameraparams.p2 = 0.0116;
cameraparams.type = 'perspectiveWithDistortion';

% Extrinsic parameters for each view
Ts{1} = [-0.0597   -0.4529    0.8896    3.3324; ...
         -0.9923    0.1242   -0.0033   -0.2495; ...
         -0.1089   -0.8829   -0.4568    1.9490; ...
          0         0         0         1.0000];
Ts{2} = [-0.0118   -0.9011   -0.4335   -1.6732; ...
         -0.9994    0.0245   -0.0236   -0.3373; ...
          0.0319    0.4330   -0.9008    0.6104; ...
          0         0         0    1.0000];
Ts{3} = [ 0.0200    0.8952    0.4451    1.7012; ...
         -0.9921    0.0728   -0.1019   -0.6151; ...
         -0.1236   -0.4396    0.8896    7.0191; ...
          0         0         0    1.0000];

figure
for view=1:3
    cameraparams.T = Ts{view};
    im = im2double(imread(['sample_data/im' num2str(view) '.jpg']));
    subplot(1,4,view)
    imshow(im)
    title(['Input view ' num2str(view)]);
    [Vxy,Vcam] = MR_project(V,cameraparams);
    [zbuffer,fbuffer,wbuffer,Vz] = MR_rasterise_mesh_mex(uint32(F), Vcam, Vxy, cameraparams.w, cameraparams.h);
    [visibility] = MR_vertex_visibility(Vxy,Vz,zbuffer,fbuffer,F);
    [samples(view).pervertexcolour,samples(view).weight] = MR_sample_image(V,F,Vxy,cameraparams,visibility,im,fbuffer,boundarydistT);    
end

pervertexcolour = MR_poisson_blending(V,F,samples,lambda);

FV.faces = F;
FV.vertices = V;
FV.facevertexcdata = pervertexcolour;

subplot(1,4,4)
patch(FV,'FaceColor','interp','EdgeColor','none'); axis equal; axis tight
title('Blended texture')