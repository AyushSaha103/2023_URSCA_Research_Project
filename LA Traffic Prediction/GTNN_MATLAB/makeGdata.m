function data = makeGdata(numFeats, batchsize, ntsteps)
    starter = zeros(batchsize, numFeats) ./ 15;
    data = zeros(batchsize * ntsteps, numFeats);
    data(1:batchsize,:) = starter;

    for b = 2:ntsteps
%         data((b-1)*batchsize+1 : (b-1)*batchsize+batchsize,:) = data((b-2)*batchsize+1 : (b-2)*batchsize+batchsize,:) + 0.03;
        data((b-1)*batchsize+1 : (b-1)*batchsize+batchsize,:) = sin(b/12);
    end

    data = rescale(data);
end

    