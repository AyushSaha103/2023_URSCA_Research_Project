classdef GTNN < handle
    % Rectangle class with width and height properties and area/perimeter methods
    properties
        ANET = handle([]);
        GNET = handle([]);
        memupdaters;
        numnodes, numfeats, numOutputFeats;

        adj, adj_pre, adjs, adjs_u;
        dcd_adjs, dcd_adj, dcdadj_pre, dadj_dadjpre;
    end
    
    methods
        function obj = GTNN(numnodes, numfeats, numOutputFeats)
            % general params
            obj.numnodes = numnodes; obj.numfeats = numfeats; obj.numOutputFeats = numOutputFeats;

            % init A-NET & G-NET
            obj.init_ANET([numfeats*2 4 1], ["relu" "sigmoid" "leaky_relu"]);
            obj.init_GNET([numfeats numOutputFeats], ["relu" "N/A"]);
            obj.init_memupdaters([2 4 4], ["relu" "sigmoid" "LSTM_custom"]);       % init memupdaters

        end

        function init_ANET(obj, layer_sizes, activations)
            obj.ANET = GTNN_NN(1, layer_sizes, activations);
            for k = obj.ANET.numLayers:-1:1         % dec. all the ANET learning rates for fair training
                obj.ANET.layers(k).lrate = obj.ANET.layers(k).lrate * 1 / obj.numnodes^(2);
            end
        end
        function init_GNET(obj, laysizes, activations)
            obj.GNET = GTNN_NN(obj.numnodes, laysizes, activations);
        end
        function init_memupdaters(obj, laysizes, activations)
            % init mem updaters
            obj.memupdaters = LSTM.empty(obj.numfeats,0);
            for f = 1:obj.numfeats
                obj.memupdaters(f) = LSTM(obj.numnodes, laysizes, activations);
            end
        end
        
        function forward(obj, state_in)
            % adj = ANET.forward(state_in)
            for n = 1:obj.numnodes
                for c = 1:obj.numnodes
                    obj.adj_pre(n,c) = obj.ANET.forward_gat( [ state_in(c,:) state_in(n,:) ] );
                end
                obj.adj(n,:) = softmax_(obj.adj_pre(n,:));
            end

            % adjs = adj * state_in
            obj.adjs = obj.adj * state_in;
            for f= 1:obj.numfeats
                % adjs_u(:,f) = memupdaters(f).ht
                obj.memupdaters(f).forward(obj.adjs(:,f));
                obj.adjs_u(:,f) = obj.memupdaters(f).ht;
            end

            % graph pred = GNET.forward(adjs_u)
            obj.GNET.forward(obj.adjs_u);
        end

        function update_allmem(obj)
            for f = 1:obj.numfeats
                obj.memupdaters(f).update_mem();
            end
        end      
        function reset_allmem(obj)
            for f = 1:obj.numfeats
                obj.memupdaters(f).reset_mem();
            end
        end
        
        function avgcost = backward(obj, state_in, strue)
            avgcost = obj.GNET.backward(strue);
            for f = 1:obj.numfeats
                obj.memupdaters(f).backward_using_deriv(obj.GNET.layers(1).dcda(:,f));
            end
            for f = 1:obj.numfeats
                obj.dcd_adjs(:,f) = obj.memupdaters(f).feedinNN.layers(1).dcda(:,2);
            end
            obj.dcd_adj = obj.dcd_adjs * state_in';

            for n = 1:obj.numnodes
                obj.dadj_dadjpre = dsoftmax(obj.adj(n,:));
                obj.dcdadj_pre(n,:) = obj.dcd_adj(n,:) * obj.dadj_dadjpre;
                for c = 1:obj.numnodes
                    obj.ANET.backward_gat([state_in(c,:) state_in(n,:)], obj.adj_pre(n,c), obj.dcdadj_pre(n,c));
                end
            end
        end

        function avgcost = pred_tplus(obj, sin, strue, nsteps)
            for tp = 1:nsteps
                obj.forward(sin);
                sin = obj.GNET.layers(end).a;
                obj.update_allmem();
            end
            disp([sin strue]);
            avgcost = mean(obj.GNET.layers(end).a(:) - strue(:));
            disp(avgcost);
        end

        function test(obj, xdata, nsteps)
            numB = floor ( height(xdata) / obj.numnodes );
            numSeq = floor ( numB / nsteps) ;
            for k = 1:numSeq - 1
                obj.pred_tplus(getbatch(k*nsteps, obj.numnodes, xdata), getbatch((k+1)*nsteps, obj.numnodes, xdata), nsteps);
            end
        end

        function train(obj, nepochs, xdata)
            numB = floor ( height(xdata) / obj.numnodes ) ;
            for k = 1:nepochs
                obj.reset_allmem();
                for b = 1:numB-1
                    s_in = xdata((b-1)*obj.numnodes + 1: (b-1)*obj.numnodes + obj.numnodes, :);
                    s_true = xdata((b)*obj.numnodes + 1: (b)*obj.numnodes + obj.numnodes, :);

                    obj.forward( s_in );
                    avgcost = obj.backward( s_in, s_true );
                    if randi(500) == 1
                        disp([ obj.GNET.layers(end).a s_true ]);
                        disp(avgcost);
                    end
                    obj.update_allmem();
                end
            end
        end


        
    end
end
