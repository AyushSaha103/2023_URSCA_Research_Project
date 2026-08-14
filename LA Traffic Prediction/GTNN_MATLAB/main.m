% 
% %% NN example
% n = NN(10,[5,7,1],["tanh","tanh", "N/A"], 0.001);
% ain = rand(10,5);
% 
% nepochs = 5000;
% for k = 1:nepochs
%     n.forward(ain);
%     cost = n.backward(ain(:,1));
%     if (randi(42) == 1)
%         disp(cost);
%     end
% end
% 
% % LSTM example
% batchsize = 11;
% ntsteps = 500;
% numFeats = 1;
% lstm = LSTM(batchsize,[2,3,4],["relu","sigmoid", "LSTM_CUSTOM"]);
% xdata = makeGdata(numFeats, batchsize, ntsteps);
% 
% nepochs = 500000;
% lstm.train(nepochs,xdata);

% GTNN example
numnodes = 2;
numfeats = 3;
numtsteps = 800;
gtnn = GTNN(numnodes, numfeats, numfeats);
xdata = makeGdata(numfeats, numnodes, numtsteps);

nepochs = 5000;
gtnn.train(nepochs,xdata);

% load('savedGTNN.mat');
gtnn.test(xdata,9);
save('savedGTNN.mat', 'gtnn');


% nsteps_pred  = 4;
% numtsteps = 18;
% xdata = makeGdata(numfeats, numnodes, numtsteps);
% load('savedGTNN.mat');
% gtnn.test(xdata,nsteps_pred);
