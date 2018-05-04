function y = fft2c(x)
n1 = size(x,1);
n2 = size(x,2);
% [nx,ny,nb] = size(x);
% y = zeros(nx,ny,nb);
% for k = 1:size(x,3)
%     y(:,:,k) = 1/sqrt(nx*ny)*fftshift(fft2(fftshift(squeeze(x(:,:,k)))));
% end
y = 1/sqrt(n1*n2)*fftshift(fftshift(fft2(fftshift(fftshift(x,1),2)),1),2);
end