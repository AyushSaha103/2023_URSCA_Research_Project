% 
% % vertical vec version
% function resvec = softmax_(vec)
%     esum = 0;
%     resvec = zeros(numel(vec), 1);
%     for i = 1:numel(vec)
%         esum = esum + exp(vec(i));
%     end
%     for i = 1:numel(vec)
%         resvec(i) = exp(vec(i)) / esum;
%     end
% end

% horizontal vec version
function resvec = softmax_(vec)
    esum = 0;
    resvec = zeros(1, numel(vec));
    for i = 1:numel(vec)
        esum = esum + exp(vec(i));
    end
    for i = 1:numel(vec)
        resvec(i) = exp(vec(i)) / esum;
    end
end
