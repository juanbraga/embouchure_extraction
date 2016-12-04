function [ ste_tc, ste_tc_norm, t_ste, x_thr_ste ] = short_time_energy( x, winlen, hop, fs, e_thr )
%ENERGIA_TIEMPO_CORTO Summary of this function goes here
%   Detailed explanation goes here

t_len = floor((length(x)-winlen)/hop);
t_ste = ((0:1:t_len-1)*(hop/fs))+(winlen/(2*fs));

slices=zeros(winlen,t_len);

for i = 0:t_len-1
   slices(:,i+1)=x(i*hop+1:i*hop+winlen).*hamming(winlen);
end

squared_slices = slices.^2;
ste_tc = sum(squared_slices)/size(squared_slices,1);

ste_tc_norm=ste_tc/max(ste_tc);
x_thr_ste=zeros(size(x));

for i = 0:t_len-1
   if ste_tc_norm(i+1)>e_thr
       x_thr_ste(i*hop+1:i*hop+winlen)=x_thr_ste(i*hop+1:i*hop+winlen)+slices(:,i+1);
   else
       x_thr_ste(i*hop+1:i*hop+winlen)=x_thr_ste(i*hop+1:i*hop+winlen); 
   end
end

% if (t_len-1)*hop+winlen < length(x)
%    t_e(end+1) = t_len*(hop/fs) + winlen/(2*fs);
%    ste_tc(end+1) = sum(x((t_len-1)*hop+winlen+1:end).^2)/length(x((t_len-1)*hop+winlen+1:end));
% 
% end

end

