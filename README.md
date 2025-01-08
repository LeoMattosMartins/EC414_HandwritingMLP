# EC414_HandwritingMLP

Neural Net MLP to distinguish between handwritten digits 0-9.

The input should have shape [B,784], where B is a batch size.
The first hidden layer a1 has 256 neurons. The second hidden layer a2 has 128 neurons. The output layer a3 has 10 neurons.
So, the output of the network with input shape [B,784] should have shape [B,10].
The weight matrices are stored in self.W1, self.W2, and self.W3.
The activation function for layers 1 and 2 is the ReLU function ReLU(z) = max(0,z). The last layer uses a different activation function, the sigmoid function, with sigmoid(z)=1/(1+exp(-z)). Thus, a3 should be a vector all of whose coordinates are between 0 and 1.
The loss function is the squared loss. You will be comparing a3 (which has size 10 for each example image) to the 1-hot vector containing all zeros except for a single 1 in the coordinate identifying the class of the image. So the loss for a single example (i.e. batch size 1) will be 0.5 * (a3 -y) **2
