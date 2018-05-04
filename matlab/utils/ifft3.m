function out = ifft3(in)
    out = ifft(ifft(ifft(in,[],1),[],2),[],3);