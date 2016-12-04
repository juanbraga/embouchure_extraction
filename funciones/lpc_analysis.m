function [ak , e_rms , e_rms_norm] = lpc_analysis(s, p)
% Calculo de la matriz de autocorrelacion 

r = zeros (p+1,1) ; 
for k = 1 : p + 1 
    r(k) = s (1: end-k+1)' * s(k : end) ; 
end

% Calculo de los coeficientes LPC
ak = toeplitz(r (1: p) )\r (2: p+1); 
e_rms = r (1)-ak' * r (2: p+1); 
e_rms_norm = e_rms/r (1) ;

end