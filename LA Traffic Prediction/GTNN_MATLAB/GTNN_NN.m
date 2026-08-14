classdef GTNN_NN < NN
    properties
    end
    
    methods
        function obj = GTNN_NN(batchsize, laysizes, activations)
            obj@NN(batchsize, laysizes, activations, 0.001);
        end

        function aidx = forward_gat(obj, sconcat)
            % aidx = forward(sconcat)
            obj.forward(sconcat);
            % aidx = leaky_relu(aidx)
            aidx = obj.layers(end).a;
            if aidx < 0
                aidx = aidx * 0.3;
            end
        end

        function backward_gat(obj, sconcat_in, aidx_pred, dcda_last_layer)
            obj.layers(end).dcda = dcda_last_layer;
            if obj.activations(end) == "leaky_relu" && aidx_pred < 0
                obj.layers(end).dcda = dcda_last_layer * 0.3;
            end
            obj.layers(1).a = sconcat_in;

            for k = obj.numLayers - 1:-1:1
                obj.layers(k).backrun();
            end
        end
        
    end
end
