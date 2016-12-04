function [ zcr_tc, zcr_tc_norm, t_zcr, x_thr_zcr ] = zero_crossing_rate( x, winlen, hop, fs, zcr_thr )
%CRUCESXCERO_TIEMPO_CORTO Summary of this function goes here
%   Detailed explanation goes here

t_len = floor((length(x)-winlen)/hop);
t_zcr = ((0:1:t_len-1)*(hop/fs))+(winlen/(2*fs));

slices=zeros(winlen,t_len);

for i = 0:t_len-1
   slices(:,i+1)=x(i*hop+1:i*hop+winlen).*hamming(winlen);
end

zcr_slices=zeros(size(slices));
zcr_tc=zeros(1,size(slices,2));

for j = 1:size(slices,2)
    for i = 1:size(slices,1)-1
        zcr_slices(i+1,j) = abs(sign(slices(i+1,j))-sign(slices(i,j)));        
    end
    zcr_tc(j) = sum(zcr_slices(:,j))/2*size(zcr_slices,1);
end

zcr_tc_norm=zcr_tc/max(zcr_tc);
x_thr_zcr=zeros(size(x));

for i = 0:t_len-1
   if zcr_tc_norm(i+1)>zcr_thr
       x_thr_zcr(i*hop+1:i*hop+winlen)=x_thr_zcr(i*hop+1:i*hop+winlen)+slices(:,i+1);
   else
       x_thr_zcr(i*hop+1:i*hop+winlen)=x_thr_zcr(i*hop+1:i*hop+winlen); 
   end
end

end

