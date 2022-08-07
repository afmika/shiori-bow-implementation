const fs = require ('fs');
const ShioriNLP = require('./ShioriNLP');
const Utils = require('./Utils');
const EPSILON = 10e-6;
const tf = require('@tensorflow/tfjs');
const Mat = require('./Mat');
const W2VSkipGramModel = require('./W2VSkipGramModel');

class WordVector {
    /**
     * @param {number[]} components 
     */
    constructor (components) {
        this.components = components || [];
    }

    /**
     * @returns number
     */
    dim () {
        return this.components.length;
    }

    /**
     * @param {number} value 
     */
    addComponent (value) {
        this.components.push (value);
    }

    /**
     * @param {WordVector} word2vec 
     */
    equals (wordvec) {
        WordVector.dimCheck (this, wordvec);
        for (let i = 0; i < this.components.length; i++)
            if (Math.abs (wordvec.components[i] - this.components[i]) > EPSILON)
                return false;
        return true;
    }

    /**
     * @param {WordVector} wordvec 
     * @returns number
     */
    dot (wordvec) {
        WordVector.dimCheck (this, wordvec);
        let s = 0;
        for (let i = 0; i < this.components.length; i++)
            s += this.components[i] * wordvec.components[i];
        return s;
    }

    /**
     * @returns number
     */
     lengthSquared () {
        return this.dot (this);
    }
    
    /**
     * @returns number
     */
     length () {
        return Math.sqrt (this.lengthSquared ());
    }

    /**
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static add (a, b) {
        WordVector.dimCheck (a, b);
        const result = new WordVector ();
        for (let i = 0; i < a.components.length; i++)
            result.addComponent (a.components[i] + b.components[i]);
        return result;
    }

    /**
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static sub (a, b) {
        WordVector.dimCheck (a, b);
        const result = new WordVector ();
        for (let i = 0; i < a.components.length; i++)
            result.addComponent (a.components[i] - b.components[i]);
        return result;
    }

    /**
     * @param {WordVector} vec 
     * @param {number} factor 
     */
    static scale (vec, factor) {
        const result = new WordVector ();
        for (let i = 0; i < vec.components.length; i++)
            result.addComponent (vec.components[i] * factor);
        return result;
    }

    /**
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static mean (a, b) {
        WordVector.dimCheck (a, b);
        const result = new WordVector ();
        for (let i = 0; i < a.components.length; i++)
            result.addComponent ((a.components[i] + b.components[i]) / 2);
        return result;
    }

    /**
     * * dot(a, b) = |a||b| * cos(angle(a, b))
     * * costDist := cost(t) 
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static cosDist (a, b) {
        let div = (a.length() * b.length());
        // well.. technically if dot(a, b) is also 0 then it should be undefined but... yeah
        if (div == 0) 
            return 0;
        return a.dot(b) / div;
    }

    /**
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static dimCheck (a, b) {
        if (a.dim() != b.dim())
            throw Error ('Operands do not share the same dimensionality');
    }

    toString () {
        return `WordVector [${this.components.join(', ')}]`;
    }
}

class ShioriWord2Vec {
    
    /**
     * @param {number?} max_vec_dimension 
     */
    constructor (max_vec_dimension = Infinity) {
        this.max_vec_dimension = max_vec_dimension;
        this.is_trained_model = false;
        this.tokens = [];
        this.vocabulary_obj = {
            word_to_idx : {},
            idx_to_word : {},
            count : 0
        };
        this.model = null;
        this.model_loss = Infinity;
    }

    /**
     * @param {string} filename 
     * @param {string} exclude_list_filename 
     */
    loadTextFromFile (filename, exclude_list_filename = null) {
        const text_content = fs
                    .readFileSync (filename)
                    .toString();
        
        this.tokens = ShioriNLP.tokenize (text_content.toLowerCase());

        if (exclude_list_filename) {
            const excluded_content = fs
                        .readFileSync (exclude_list_filename)
                        .toString();
            this.excluded_tokens = ShioriNLP.tokenize (excluded_content.toLowerCase());
            this.tokens = this.tokens.filter (it => !this.excluded_tokens.includes(it));
        }
        
        // reset
        this.vocabulary_obj = {
            word_to_idx : {},
            idx_to_word : {},
            count : 0
        };

        let index = 0;
        for (let i = 0; i < this.tokens.length; i++) {
            const token = this.tokens[i];
            if (this.vocabulary_obj.word_to_idx[token] == undefined) {
                this.vocabulary_obj.word_to_idx[token] = index;
                this.vocabulary_obj.idx_to_word[index] = token;
                this.vocabulary_obj.count++;
                index++;

            }
        }
    }

    /**
     * @param {string} input_word 
     * @returns {Mat} hot encoded col vector
     */
    hotEncode (input_word) {
        const hot_vec = [];
        for (let word in this.vocabulary_obj.word_to_idx) {
            let index = this.vocabulary_obj.word_to_idx[word];
            hot_vec[index] = input_word == word ? 1 : 0;
        }
        return Mat.vec(...hot_vec);
    }

    /**
     * @param {number} n_context 
     * @returns 
     */
    generateTrainingDatas (n_context = 1) {
        let tokens = this.tokens;
        const isValidIndex = x => x >= 0 && x < tokens.length;

        const datas = [];
        for (let cursor = 0; cursor < tokens.length; cursor++) {
            const center_word = {
                word : tokens[cursor],
                vec : this.hotEncode(tokens[cursor])
            };
            const context_words = [];
            for (let j = -n_context; j <= n_context; j++) {
                let index = cursor + j;
                if (isValidIndex(index) && index != cursor)
                    context_words.push({
                        word : tokens[index],
                        vec : this.hotEncode(tokens[index])
                    });
            }
            datas.push ({
                token : tokens[cursor],
                center : center_word,
                context_list : context_words
            });
        }

        return datas;
    }

    /**
     * * Tries its best to reduce the number of vector columns while retaining the relevant ones
     * * The number of columns will be equal to min(computed_col_length, max_vec_dimension)
     * @param {number?} n_context number of 'context' word on the left/right of a given token
     * @param {Function?} log_fun Function (message, n_current, total)
     */
    async trainOptimally (n_context = 1, epochs = 50, log_fun = null) {
        const n_input = this.vocabulary_obj.count;
        const n_output = this.vocabulary_obj.count;

        const desired_output_dim = 100; // we can put whatever we want
        this.model = new W2VSkipGramModel(desired_output_dim, n_output);
        console.log('loading inputs');
        const dataset = this.generateTrainingDatas();
        console.log('begin', dataset.length);

        // ex : w1 x w2 w3
        // input  := x  x  x (we repeat x as many times as there are context words)
        // output := w1 w2 w3
        for (let i = 0; i < epochs; i++) {
            let epoch_loss = 0;
            for (const {center, context_list} of dataset) {
                const input = center.vec;
                const w_idx = this.vocabulary_obj.word_to_idx[center.word];
                const {
                        output_yj, 
                        output_h, 
                        output_u
                    } = this.model.optimisedFeedforward (w_idx);

                // const {
                //         output_yj, 
                //         output_h, 
                //         output_u
                //     } = this.model.feedforward (input);
                
                // error vector
                const zeroes = new Array(n_output).fill(0);
                let El = Mat.vec(... zeroes);

                let sum_ujc_star = 0;
                for (let context of context_list) {
                    const context_vec = context.vec;
                    const ejc = context_vec.map((tc, i, j) => output_yj - tc);
                    El = El.add(ejc);
                    // improvised indexOf
                    // const context_indexer = context_vec.transpose();
                    // sum_ujc_star += context_indexer.prod(output_u).get(0, 0);
                    const jc_star = this.vocabulary_obj.word_to_idx[context.word];
                    const temp = output_u.get(jc_star, 0);
                    sum_ujc_star += temp;
                }

                // improvised indexOf
                // Sum exp(uk)
                const fold_exp_sum = (dsum, uk) => dsum + Math.exp(uk);
                const sum_exp_uk = output_u.foldToScalar(fold_exp_sum);
                const C = context_list.length;

                // this.model.h_weights.print();

                const current_loss = -sum_ujc_star + C * Math.log(sum_exp_uk);
                epoch_loss += current_loss;

                
                // output layer raw outputs u do not participate in the backprop
                // only the error matters
                this.model.backprop (El, output_h, input);
            }

            this.model_loss = epoch_loss;
            Utils.safeRun(log_fun) (this.model_loss, i + 1, epochs);
        }

        const words = {}; 
        for (let {token} of dataset) {
            const vec = this.hotEncode (token);
            const {output_h} = this.model.feedforward(vec);
            words[token] = new WordVector(output_h.transpose().entries[0]);
        }

        this.is_trained_model = true;
        this.words = words;
    }

    saveVectorsTo (filename) {
        const text = JSON.stringify ({
            metadata : {
                generator : 'ShioriWord2Vec',
                infos : this.infos()
            },
            words : this.words
        });

        fs.writeFileSync (filename, text);
    }

    /**
     * @param {string} filename 
     */
    loadVectorsFromFile (filename) {
        const datas = fs.readFileSync (filename).toString();
        const {words} = JSON.parse (datas);
        if (!words)
            throw Error ("Error data format, 'words' field expected");

        for (let word in words) {
            const current = words[word];
            if (!current.components)
                throw Error ("Error data format at word '" + word + "', 'components' field expected");
            // wrap each json object inside a WordVector object
            words[word] = new WordVector (current.components);
        }

        this.words = words;
        this.is_trained_model = true;
    }

    /**
     * @param {string} word 
     * @returns {WordVector}
     */
    word2vec (word) {
        word = word.toLowerCase ();
        if (!this.is_trained_model)
            throw Error ('Please train the model first !');
        if (this.words[word] == undefined)
            throw Error (word + ' is not a part of the dataset !');
        return this.words[word];
    }

    /**
     * @param {string} word 
     * @param {number?} top 
     * @returns [{word, dist, cosine_dist}] 
     */
    closestWord (word, top = 3) {
        const vector = this.word2vec(word);
        return this.closestWordByVector (vector, top, word.toLowerCase ());
    }

    /**
     * @param {string} word_a 
     * @param {string} word_b 
     * @param {number} top 
     */
    meanWord (word_a, word_b, top = 3) {
        const vector_a = this.word2vec (word_a);
        const vector_b = this.word2vec (word_b);
        const mean = WordVector.mean (vector_a, vector_b);
        return this.closestWordByVector (mean, top);
    }

    
    /**
     * @param {WordVector} vector 
     * @param {number?} top 
     * @param {string?} trivial_word 
     * @returns [{word, dist, cosine_dist}]
     */
    closestWordByVector (vector, top = 3, trivial_word = null) {
        let top_list = [];
        for (let item in this.words) {
            if (trivial_word == item) continue;
            const vec = this.words[item];
            const dist = WordVector.sub (vec, vector).length();
            const cosine_dist = WordVector.cosDist (vec, vector);
            top_list.push({word : item, cosine_dist : cosine_dist, dist : dist, vec : vec});
        }

        top_list
            .sort((a, b) => {
                // a < b
                let dist_diff = a.dist - b.dist;
                // we do not care about the orientation, we just care
                // about the angle between the two vectors
                // the bigger the cosine diff, the more similar the vectors are !
                // b > a
                let cos_diff = b.cosine_dist - a.cosine_dist;
                // prioritize the cosine dist
                let same_cos_dist = Math.abs(1 - cos_diff) < EPSILON;
                return same_cos_dist ? dist_diff : cos_diff;
            });

        return top_list.filter((_, i) => i < top);
    }

    /**
     * @returns infos
     */
    infos () {
        const vec_dim = this.words [ Object.keys(this.words) [0] ].dim();
        return {
            total_words : Object.keys(this.words).length,
            vec_dim : vec_dim
        };
    }                                                                                                                                                                                                                                                                                                                        
}


module.exports = {
    ShioriWord2Vec : ShioriWord2Vec,
    WordVector : WordVector
};