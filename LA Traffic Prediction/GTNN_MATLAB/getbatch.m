function batch = getbatch(bidx, batchsize, xdata)
    batch = xdata((bidx-1)*batchsize + 1: (bidx-1)*batchsize + batchsize, :);
end