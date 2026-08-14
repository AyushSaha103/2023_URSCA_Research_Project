

classdef LSTM < handle
    % Rectangle class with width and height properties and area/perimeter methods
    properties
        feedinNN = handle([]);
        batchsize;

        ct, ct_1, ht, ht_1;
    end
    
    methods
        function obj = LSTM(batchsize, feedin_laysizes, feedin_activations)
            obj.batchsize = batchsize;
            obj.feedinNN = LSTM_NN(batchsize, feedin_laysizes, feedin_activations);
            obj.reset_mem();
        end
        function forward(obj, xt)
            obj.feedinNN.forwardfeed([obj.ht_1 xt]);        % result = ABCD = feedinNN.<lastlayer>.a
            lastl = obj.feedinNN.layers(end);

            obj.ct = (obj.ct_1 .* lastl.a(:,1)) + (lastl.a(:,2) .* lastl.a(:,3));      % ct = (ct_1 .* A) + (B .* C)
            obj.ht = tanh(obj.ct) .* lastl.a(:,4);                                  % ht = tanh(ct) .* D
        end

        function avgcost = backward(obj, ht_true)
            dcdht = 2 * (obj.ht - ht_true);
            avgcost = -0.5 * mean(dcdht);
            lastl = obj.feedinNN.layers(end);

            % set last feedinNN layer dcda
            lastl.dcda(:,4) = dcdht .* tanh(obj.ct);
            dcdct = dcdht .* lastl.a(:,4);
            lastl.dcda(:,1) = dcdct .* obj.ct_1;
            lastl.dcda(:,2) = dcdct .* lastl.a(:,3);
            lastl.dcda(:,3) = dcdct .*  lastl.a(:,2);

            %----------------
            % apply changes to ht, ct
            obj.ht = obj.ht - 0.001 * dcdht;
            obj.ct = obj.ct - 0.001 * dcdct;
            %----------------

            obj.feedinNN.backwardfeed();
        end



        function train(obj, nepochs, xdata)

            numR = height(xdata);
            numB = floor(numR / obj.batchsize);
            for k = 1:nepochs
                obj.reset_mem();
                for b = 1:numB-1
                    xt = xdata((b-1)*obj.batchsize + 1: (b-1)*obj.batchsize + obj.batchsize,   1   );
                    ht_true = xdata((b)*obj.batchsize + 1: (b)*obj.batchsize+ obj.batchsize,    1    );
                    obj.forward(xt);
                    avgcost = obj.backward(ht_true);
                    if (randi(500) == 1)
                        disp([ ht_true obj.ht]);
                        disp(avgcost);
                    end
                    obj.update_mem();
                end
            end
        end


        % MEM UPDATES
        function reset_mem(obj)
            obj.ht_1 = zeros(obj.batchsize,1);
            obj.ct_1 = zeros(obj.batchsize,1);
        end
        function update_mem(obj)
            obj.ht_1 = obj.ht;
            obj.ct_1 = obj.ct;
        end

        %% ADDED funcs to support GTNN
        function avgcost = backward_using_deriv(obj, dcdht)
            avgcost = -0.5 * mean(dcdht);
            lastl = obj.feedinNN.layers(end);

            % set last feedinNN layer dcda
            lastl.dcda(:,4) = dcdht .* tanh(obj.ct);
            dcdct = dcdht .* lastl.a(:,4);
            lastl.dcda(:,1) = dcdct .* obj.ct_1;
            lastl.dcda(:,2) = dcdct .* lastl.a(:,3);
            lastl.dcda(:,3) = dcdct .*  lastl.a(:,2);

            %----------------
            % apply changes to ht, ct
            obj.ht = obj.ht - 0.001 * dcdht;
            obj.ct = obj.ct - 0.001 * dcdct;
            %----------------

            obj.feedinNN.backwardfeed();
        end
    end
end
