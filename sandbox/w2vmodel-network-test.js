const Mat = require("../core/Mat");
const W2VSkipGramModel = require("../core/W2VSkipGramModel");

const V = 9; // vocab size
const N = 10; // desired resulting word vector (= nb of hidden layer weight col)

const model = new W2VSkipGramModel (2, 3);
console.log(model.h_weights);
console.log(model.o_weights);

const input = Mat.vec(0, 1, 0);
const target = Mat.vec(1, 0, 0);
const {output_yj, output_h, output_u} = model.feedforward(input);
const error = output_u.sub(input);

console.log('# before');
model.o_weights.print();

model.backprop (error, output_h, input);

console.log('# after');
model.o_weights.print();


const test = model.softmax(Mat.vec( 0.53418708, -0.47742903, -0.7970129));
test.print();
console.log(test.foldToScalar((acc, x) => acc + x))