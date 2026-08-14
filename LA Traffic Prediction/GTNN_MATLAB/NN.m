

classdef NN < handle
    % Rectangle class with width and height properties and area/perimeter methods
    properties
        layers;
        layersizes, activations;
        numLayers, batchsize;
    end
    
    methods
        function obj = NN(batchsize, layersizes, activations, lrate)
            % general params
            obj.numLayers = numel(layersizes);
            obj.batchsize = batchsize;
            obj.layersizes = layersizes;
            obj.activations = activations;

            % declare layers
            obj.layers = [];
            obj.layers = NNlayer.empty(obj.numLayers,0);

            % init. and link layers
            for l = 1:obj.numLayers
                obj.layers(l) = NNlayer(batchsize, layersizes(l), lrate); obj.layers(l).activation = activations(l);
                if l > 1
                    obj.layers(l-1).link(obj.layers(l));
                end
            end
        end

        function pred = forward(obj, ain)
            obj.layers(1).a = ain;
            for k = 1:obj.numLayers - 1
                obj.layers(k).forwardrun();
            end
            pred = obj.layers(end).a;
        end

        function avgcost = backward(obj, yin)
            obj.layers(end).dcda = 2 * (obj.layers(end).a - yin);
            avgcost = -0.5 * mean( obj.layers(end).dcda(:) );

            for k = obj.numLayers - 1:-1:1
                obj.layers(k).backrun();
            end
        end

        function backward_given_dcda(obj, dcda_in)
            obj.layers(end).dcda = dcda_in;
%             avgcost = -0.5 * mean( obj.layers(end).dcda(:) );

            for k = obj.numLayers - 1:-1:1
                obj.layers(k).backrun();
            end
        end



    end
end
