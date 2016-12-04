function [ gt ] = get_ground_truth( emb_gt, t_slices )
%GET_GROUND_TRUTH Summary of this function goes here
%   Detailed explanation goes here

gt = zeros(size(t_slices));

for i=1:length(emb_gt)-1
   gt( (t_slices>=emb_gt(i,1))&(t_slices<emb_gt(i+1,1)) )= emb_gt(i,2);
end

end

