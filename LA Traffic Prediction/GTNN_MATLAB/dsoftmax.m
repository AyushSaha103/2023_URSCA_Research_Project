function res = dsoftmax(vec)
    res = zeros(numel(vec), numel(vec));

    for i = 1:numel(vec)
        for j = 1:numel(vec)
            if i == j
                res(i,j) = vec(i) * (1 - vec(i));
            else
                res(i,j) = -vec(i) * vec(j);
            end
        end
    end
end
