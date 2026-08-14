
function a = tanh(z)
    a = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));
end

function a = sig(z)
    a = 1 ./ (1 + exp(-z));
end

function a = relu(z)
    a = (z>0) .* z;
end

function dadz = drelu(a)
    dadz = (a > 0);
end