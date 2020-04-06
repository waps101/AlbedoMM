function [pervertextex,weight] = MR_sample_image(V,F,Vxy,cameraparams,visibility,im,fbuffer,boundarydistT)
%MR_SAMPLE_IMAGE Sample an image onto a mesh given rasterisation
%   Inputs:
%     V             - nverts x 3 matrix of vertex positions
%     F             - nfaces x 3 matrix of face indices
%     Vxy           - nverts x 2 projected 2D vertex positions
%     cameraparams  - structure containing camera parameters
%     visibility    - nverts x 1 per-vertex binary visibility
%     im            - H x W x 3 image
%     fbuffer       - face buffer from rasteriser
%     boundarydistT - distance to boundary threshold (in pixels)
%
%   Outputs:
%     pervertextex  - nverts x 3 per-vertex sampled colours
%     weight        - pervertex weight, 0 is not visible, else cosine of 
%                     angle between viewer and surface normal
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

% Compute camera centre
c = -cameraparams.T(1:3,1:3)'*cameraparams.T(1:3,4);
% Compute per-vertex view vectors
Views(:,1) = c(1)-V(:,1);
Views(:,2) = c(2)-V(:,2);
Views(:,3) = c(3)-V(:,3);
%Views = -Views;
Views = Views./repmat(sqrt(sum(Views.^2,2)),[1 3]);

% Sample image onto vertices
pervertextex(:,1) = interp2(im(:,:,1),Vxy(:,1),Vxy(:,2));
pervertextex(:,2) = interp2(im(:,:,2),Vxy(:,1),Vxy(:,2));
pervertextex(:,3) = interp2(im(:,:,3),Vxy(:,1),Vxy(:,2));

dist2boundary = bwdist(~imfill(fbuffer~=0,'holes'));
pervertexd2b = interp2(dist2boundary,Vxy(:,1),Vxy(:,2));

% Compute per vertex normals
pervertexnormals = MR_vertex_normals(F,V);

% Calculate weight as clamped dot product between normal and viewer, masked
% by visibility
weight = max(0,sum(pervertexnormals.*Views,2)).*visibility;

weight(isnan(pervertextex(:,1)))=0;

weight(pervertexd2b<boundarydistT) = 0;

pervertextex(isnan(pervertextex))=0;

end

