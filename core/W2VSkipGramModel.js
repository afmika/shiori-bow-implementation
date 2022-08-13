const Mat = require("./Mat");

module.exports = class W2VSkipGramModel {
    /**
     * @param {number} desired_vec_dim number of neurons on the hidden layer
     * @param {number} vocab_dim  number of units on the input/output
     */
    constructor (desired_vec_dim, vocab_dim) {
        this.vocab_dim = vocab_dim; // V
        this.desired_vec_dim = desired_vec_dim; // N
        this.learning_rate = 0.01;

        // V x N
        // input layer -> hidden layer weights
        const range = (r, i, j) => 2 * r - 1;
        this.h_weights = Mat.rand (this.vocab_dim, this.desired_vec_dim, range);

        // N x V
        // hidden layer -> output layer weights
        this.o_weights = Mat.rand (this.desired_vec_dim, this.vocab_dim, range);
        this.o_weights.printShape();
    }

    /**
     * Returns a vector such that the sum of all of its entry = 1
     * @param {Mat} vec
     * @returns {Mat} 
     */
    softmax (vec) {
        let max = -Infinity;
        vec.each((vi, i, j) => {
            max = Math.max(vi, max);
        });
        let sum_vec_ex = 0;
        const vec_ex = vec.map((vi, i, j) => {
            const entry = Math.exp(vi - max);
            sum_vec_ex += entry;
            return entry;
        });
        return vec_ex.map(it => it / sum_vec_ex);
    }

    /**
     * Returns a vector such that the sum of all of its entry = 1
     * @param {Mat} vec 
     * @returns 
     */
    fastSoftmax (vec) {
        const total = vec.foldToScalar ((acc, x) => acc + Math.exp(x));
        return vec.map(it => Math.exp(it) / total);
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

        // fetch j-th element of u such that input_vec[j] = 1
        // uj is a vector V x 1
        // const y = this.softmax(u);
        const y = this.fastSoftmax(u);
        return {
            // (L x 1) ^ T. (L x 1) => (1 x L) . (L x 1) => 1 x 1
            output_y : y,
            output_h : h,
            output_u : u
        };
    }

    /**
     * Reference : word2vec Parameter Learning Explained by Xin Rong
     * @param {Mat} errors column vector between the output layer and a target
     * @param {Mat} h_output column vector representing the output of the hidden layer
     * @param {Mat} input training example
     */
    backprop (errors, h_output, input) {
        const lr = this.learning_rate;

        // output -> hidden layer
        const dw_output = h_output.outerProd (errors);
        // hidden layer -> input
        // update first layer first before updating the output layer (input -> hidden)
        // const temp = this.h_weights.prod (errors.transpose());
        
        const temp = Mat.prodTransposeLeft (this.h_weights, errors);
        const dw_hidden = input.outerProd (temp);
        this.findNaN(dw_hidden, 'got dw_hidden first', errors);

        this.h_weights = this.h_weights.sub (dw_hidden.scale(lr));

        // temp = W'^T e
        this.o_weights = this.o_weights.sub (dw_output.scale(lr));
    }

    /**
     * Reference : word2vec Parameter Learning Explained by Xin Rong
     * @param {Mat} errors column vector between the output layer and a target
     * @param {Mat} h_output column vector representing the output of the hidden layer
     * @param {Mat} input training example
     */
    fastBackprop (errors, h_output, input) {
        const lr = this.learning_rate;

        // output -> hidden layer
        const dw_output = h_output.outerProd (errors);
        // hidden layer -> input
        // update first layer first before updating the output layer (input -> hidden)
        // const temp = this.h_weights.prod (errors.transpose());
        // these two computations should be parallelizable
        const temp = Mat.prodTransposeLeft (this.h_weights, errors);
        const dw_hidden = input.outerProd (temp);

        // these two computations should be parallelizable
        this.h_weights = this.h_weights.map((wij, i, j) => {
            return wij - lr * dw_hidden.get(i, j);
        });

        this.o_weights = this.o_weights.map((wij, i, j) => {
            return wij - lr * dw_output.get(i, j);
        });
    }

    findNaN(mat, debug, ...other_vec) {
        mat.each((v, i, j) => {
            if (isNaN(v)) {
                if (other_vec)
                    other_vec.forEach(v => v.print());
                throw Error ('Got NaN at ' +i+','+j +' : ' + debug)
            }
        })
    }
}