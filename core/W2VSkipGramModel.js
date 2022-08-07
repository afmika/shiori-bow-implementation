const Mat = require("./Mat");

module.exports = class W2VSkipGramModel {
    /**
     * @param {number} desired_vec_dim number of neurons on the hidden layer
     * @param {number} vocab_dim  number of units on the input/output
     */
    constructor (desired_vec_dim, vocab_dim) {
        this.vocab_dim = vocab_dim; // V
        this.desired_vec_dim = desired_vec_dim; // N
        this.learning_rate = .01;

        // V x N
        // input layer -> hidden layer weights
        const range = (r, i, j) => 2 * r - 1;
        this.h_weights = Mat.rand (this.vocab_dim, this.desired_vec_dim, range);

        // N x V
        // hidden layer -> output layer weights
        this.o_weights = Mat.rand (this.desired_vec_dim, this.vocab_dim, range);
    }

    /**
     * @param {number} one_index index
     */
     optimisedFeedforward (one_index) {
        // h = W^T x
        // const h = this.h_weights.transpose().prod(input_vec);
        
        // v = W^T x = row weight
        const h = Mat.vec(... this.h_weights.entries[one_index]);
        // u = W'^T h
        const u = Mat.prodTransposeLeft(this.o_weights, h);

        // output (using softmax)
        let ds = 0;
        u.each ((uk, row, col) => {
            ds += Math.exp(uk);
        });
        // fetch j-th element of u such that input_vec[j] = 1
        // uj is a vector V x 1
        const uj = u.get(one_index, 0);
        
        const yj = Math.exp(uj) / ds; // basically P(wj|winput)

        return {
            output_yj : yj,
            output_h : h,
            output_u : u
        };
    }



    /**
     * @param {Mat} input_mat vector
     */
    feedforward (input_vec) {
        // h = W^T x
        // const h = this.h_weights.transpose().prod(input_vec);
        const h = Mat.prodTransposeLeft(this.h_weights, input_vec);
        // u = W'^T h
        // const u = this.o_weights.transpose().prod(h);
        const u = Mat.prodTransposeLeft(this.o_weights, h);

        // output (using softmax)
        let ds = 0;
        u.each ((uk, row, col) => {
            ds += Math.exp(uk);
        });
        // fetch j-th element of u such that input_vec[j] = 1
        // uj is a vector V x 1
        // const uj = u.transpose().prod(input_vec);
        const uj = Mat.prodTransposeLeft (u, input_vec);
        
        const yj = Math.exp(uj.get(0, 0)) / ds; // basically P(wj|winput)

        return {
            output_yj : yj,
            output_h : h,
            output_u : u
        };
    }

    /**
     * Reference : word2vec Parameter Learning Explained by Xin Rong
     * @param {Mat} errors column vector between the output layer and a target
     * @param {Mat} hidden column vector representing the output of the hidden layer
     * @param {Mat} input training example
     */
    backprop (errors, hidden, input) {
        const lr = this.learning_rate;

        // output -> hidden layer
        const dw_output = hidden.outerProd (errors);
        this.o_weights = this.o_weights.sub (dw_output.scale(lr));

        // hidden layer -> input
        // temp = W'^T e
        // const temp = this.h_weights.prod (errors.transpose());
        const temp = Mat.prodTransposeLeft(this.h_weights, errors);

        const dw_hidden = input.outerProd (temp);
        this.h_weights = this.h_weights.sub (dw_hidden.scale(lr));
        // this.h_weights.printShape();
    }
}