function [ slices, t_slices ] = slice_audio( x, fs, winlen, hop )
%SLICE_AUDIO Summary of this function goes here
%   Detailed explanation goes here

t_len = floor((length(x)-winlen)/hop);
t_slices = ((0:1:t_len-1)*(hop/fs))+(winlen/(2*fs));

p=26; %numero de polos
slices=zeros(winlen,t_len);

for i = 0:t_len-1
   slices(:,i+1)=x(i*hop+1:i*hop+winlen).*hamming(winlen);
end

end

