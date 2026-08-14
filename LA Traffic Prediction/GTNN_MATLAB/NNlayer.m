

classdef NNlayer < handle
    % Rectangle class with width and height properties and area/perimeter methods
    properties
        a,w,b;
        dcda, dcdw, dcdb;
        dcdz, dadz;
        activation ="relu";
        numN, numNex=0, batchsize;
        next = handle([]);
        prev = handle([]);
        lrate = 0.001;
    end
    
    methods
        function obj = NNlayer(batchsize, numN, lrate)
            obj.batchsize = batchsize; obj.numN = numN;     % set this layer batchsize & numN
            obj.a = rand(obj.batchsize, numN);
            obj.dcda = rand(obj.batchsize, numN);
            obj.dcdz = rand(obj.batchsize, numN);
            obj.dadz = rand(obj.batchsize, numN);
            obj.lrate = lrate;
        end
        function link(obj, nextLayer)
            obj.next = nextLayer;
            nextLayer.prev = obj;

            obj.numNex = nextLayer.numN;        % set first layer numNex
%             obj.w = rand(obj.numN, obj.numNex);         % first layer w
%             obj.b = rand(obj.batchsize, obj.numNex);        % first layer b
            obj.w = zeros(obj.numN, obj.numNex);
            obj.b = zeros(obj.batchsize, obj.numNex);

        end

        function changeWB(obj)
            obj.w = obj.w - obj.lrate * obj.dcdw;
            obj.b = obj.b - obj.lrate * obj.dcdb;
        end

        function forwardrun(obj)
            z = obj.a * obj.w + obj.b;

            if obj.activation == "relu"
                obj.next.a = (z>0) .* z;
            elseif obj.activation == "sigmoid"
                obj.next.a = 1 ./ (1 + exp(-z));
            elseif obj.activation == "tanh"
                obj.next.a = (exp(z) - exp(-z)) ./ (exp(z) + exp(-z));
            else
                obj.next.a = z;
            end
        end

        function backrun(obj)
            % next.dcdz = de_activation( next a )
            if obj.activation == "relu"
                obj.next.dadz = (obj.next.a > 0);
            elseif obj.activation == "sigmoid"
                obj.next.dadz = 1 - obj.next.a;
            elseif obj.activation == "tanh"
                obj.next.dadz = 1 - obj.next.a.^2;
            end

            obj.next.dcdz = obj.next.dcda .* obj.next.dadz;     % next dcdz = next dcda .* next dadz
            obj.dcdw = obj.a' * obj.next.dcdz;                  % dcdw = a' * next dcdz
            obj.dcdb = obj.next.dcdz;                           % dcdb = next dcdz
            obj.dcda = obj.next.dcdz * obj.w';                  % dcda = next dcdz * w'

            obj.changeWB();         % change W,B
        end


        
    end
end
