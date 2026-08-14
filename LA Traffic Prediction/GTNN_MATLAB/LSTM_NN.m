



classdef LSTM_NN < NN
    % Rectangle class with width and height properties and area/perimeter methods
    properties
    end
    
    methods
        function obj = LSTM_NN(batchsize, layersizes, activations)
            obj@NN(batchsize, layersizes, activations, 0.001);
        end
        function forwardfeed(obj, hxconcat)
            obj.layers(1).a = hxconcat;
            for k = 1:obj.numLayers-2
                obj.layers(k).forwardrun();
            end

            % CUSTOM forward run into last layer
            pre_lastl = obj.layers(end-1);
            lastl = obj.layers(end);
            z = pre_lastl.a * pre_lastl.w + pre_lastl.b;
            lastl.a(:,1) = sig(z(:,1));
            lastl.a(:,2) = sig(z(:,2));
            lastl.a(:,3) = tanh(z(:,3));
            lastl.a(:,4) = sig(z(:,4));
        end
        function backwardfeed(obj)
            lastl = obj.layers(end);        % last layer
            pre_lastl = obj.layers(end-1);      % prev of last layer

            lastl.dadz(:,1) = 1 - lastl.a(:,1);
            lastl.dadz(:,2) = 1 - lastl.a(:,2);
            lastl.dadz(:,3) = 1 - lastl.a(:,3).^2;
            lastl.dadz(:,4) = 1 - lastl.a(:,4);

            lastl.dcdz = lastl.dcda .* lastl.dadz;
            pre_lastl.dcdw = pre_lastl.a' * lastl.dcdz;
            pre_lastl.dcdb = lastl.dcdz;
            pre_lastl.dcda = lastl.dcdz * pre_lastl.w';
            pre_lastl.changeWB();

            for k = obj.numLayers-2:-1:1
                obj.layers(k).backrun();
            end
                
        end
    end

end

