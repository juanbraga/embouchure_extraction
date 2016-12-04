function [ voicing, t_voicing ] = voicing( x, winlen, hop, fs, num_lags, kappa )

voicing_len=floor((length(x)-(winlen+num_lags))/hop);
t_voicing=((0:1:voicing_len-1)*(hop/fs))+((winlen+num_lags)/(2*fs));

slices=zeros(winlen+num_lags,voicing_len);
d=zeros(num_lags,voicing_len);
d_prime=zeros(num_lags,voicing_len);

for i = 0:voicing_len-1
   slices(:,i+1)=x(i*hop+1:i*hop+(winlen+num_lags));
   for j=0:num_lags-1
        if j==0
            d_prime(1,i+1)=1;
        else
            d(j+1,i+1) = sum((slices(1:winlen,i+1)-slices(1+j:winlen+j,i+1)).^2);
            d_prime(j+1,i+1) = d(j+1,i+1)*j/sum(d(1:j+1,i+1));
        end
   end
end
%%
% figure, imagesc(t_voicing, 0:num_lags-1, d_prime), 
% colorbar('EastOutside'), axis xy;

d_prime_threshold=d_prime;
d_prime_threshold(d_prime>kappa)=0;

% figure, imagesc(t_voicing, 0:num_lags-1, d_prime_threshold), 
% colorbar('EastOutside'), axis xy;

voicing=zeros(1,voicing_len);
locs=zeros(1,voicing_len);

for i = 1:voicing_len
   [pks_aux,locs_aux]=findpeaks(-d_prime_threshold(:,i),'NPeaks',1);
   if isempty(pks_aux)
       [voicing(i), locs(i)]=min(d_prime(:,i));
   else
       locs(i)=locs_aux;
       voicing(i)=d_prime(locs_aux,i);
   end
end

voicing=1-voicing;


