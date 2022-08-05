const Mat = require("../core/Mat");
const W2VSkipGramModel = require("../core/W2VSkipGramModel");

const V = 9; // vocab size
const N = 10; // desired resulting word vector (= nb of hidden layer weight col)

const model = new W2VSkipGramModel (2, 3);
console.log(model.h_weights);
console.log(model.o_weights);
const output = model.feedforward(Mat.vec(0, 1, 0));
console.log(output);