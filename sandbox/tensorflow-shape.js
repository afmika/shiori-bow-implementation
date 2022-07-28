const tf = require('@tensorflow/tfjs');

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
// require('@tensorflow/tfjs-node');

// Train a simple model:


// const xs = tf.randomNormal([100, 10]);
// const ys = tf.randomNormal([100, 1]);
const inp_n_row = 4;
const inp_n_col = 1;

const model = tf.sequential();

// note
// units : number of neurons
// activation : activation function
// inputShape : shape of the input (reshape logic if we use an array with 3 values)


// dense is a fully connected layer
// use relu for the input
const hidden = tf.layers.dense({units: 50, activation: 'relu', inputShape: [2]});

// use the sigmoid for the output layer
const output = tf.layers.dense({units: 1, activation: 'sigmoid'});

model.add(hidden);
model.add(output);

// sgdOpt
// we can also use adam but let's stick with the gradient descent
const sgdOpt = tf.train.sgd(0.1); // learning rate
// model.compile({optimizer: sgdOpt, loss: 'meanSquaredError'});
// works much more faster for binary-ish values
model.compile({optimizer: sgdOpt, loss: 'binaryCrossentropy'});

const input = tf.tensor2d([[1, 0], [0, 1], [1, 1], [0, 0]]);
const label = tf.tensor2d([[1], [1], [0], [0]]);



(async () => {

    for (let i = 0; i < 200; i++) {
        const history = await model.fit(input, label);
    }
    
    // prediction
    const inputs_to_pred = [
        [1, 1], 
        [0, 0],
        [1, 0],
        [0, 1]
    ];
    const pred = model.predict(
        tf.tensor2d(inputs_to_pred)
    );

    pred.print();
}) ();


