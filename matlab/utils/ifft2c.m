function y = ifft2c(x)
n1 = size(x,1);
n2 = size(x,2);
% [nx,ny,nb] = size(x);
% y = zeros(nx,ny,nb);
% for k = 1:size(x,3)
%     y(:,:,k) = sqrt(nx*ny)*ifftshift(ifft2(ifftshift(squeeze(x(:,:,k)))));
% end
y = sqrt(n1*n2)*ifftshift(ifftshift(ifft2(ifftshift(ifftshift(x,1),2)),1),2);

end