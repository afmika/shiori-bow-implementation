const Mat = require("./Mat");

module.exports = class W2VSkipGramModel {
    /**
     * @param {number} desired_vec_dim number of neurons on the hidden layer
     * @param {number} vocab_dim  number of units on the input/output
     */
    constructor (desired_vec_dim, vocab_dim) {
        this.vocab_dim = vocab_dim; // V
        this.desired_vec_dim = desired_vec_dim; // N

        // V x N
        // input layer -> hidden layer weights
        const range = (r, i, j) => i + j;
        this.h_weights = Mat.rand (this.vocab_dim, this.desired_vec_dim, range);

        // N x V
        // hidden layer -> output layer weights
        this.o_weights = Mat.rand (this.desired_vec_dim, this.vocab_dim, range);
    }


    /**
     * @param {Mat} input_mat vector
     */
    feedforward (input_vec) {
        // h = W^T x
        const h = this.h_weights.transpose().prod(input_vec);
        // u = W'^T h
        const u = this.o_weights.transpose().prod(h);

        // output (using softmax)
        let ds = 0;
        u.each ((uk, row, col) => {
            ds += Math.exp(uk);
        });
        // fetch j-th element of u such that input_vec[j] = 1
        // uj is a vector V x 1
        const uj = u.transpose().prod(input_vec);
        const yj = Math.exp(uj.get(0, 0)) / ds; // basically P(wj|winput)

        return {
            output_yj : yj,
            output_h : h,
            output_u : u
        };
    }

    backprop () {

    }
}