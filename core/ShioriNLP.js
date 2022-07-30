const tf = require('@tensorflow/tfjs');
const fs = require('fs');


module.exports = class ShioriNLP {
    constructor () {
        this.intents = [];
        this.intent_func = {};
        this.vocabulary = [];
        this.consider_word_order = false;
        this.model = null;
        this.model_loss = Infinity;
    }

    /**
     * @param {string} filename 
     */
    load (filename) {
        const intents_array = JSON.parse(
            fs.readFileSync(filename)
        );
        const seen = new Set();
        for (let intent of intents_array) {
            // "tag": "shop",
            // "patterns": ["Id like to buy something", "whats on the menu", "what do you reccommend?", "could i get something to eat"],
            // "responses": ["We sell chocolate chip cookies for $2!", "Cookies are on the menu!"],
            // "context_set": "after_context_1_enabled_idk"

            this.intents.push({
                tag : intent.tag,
                patterns : intent.patterns, // questions array
                responses : intent.responses, // fallback responses array
                context_set : intent.context_set // context rule
            });

            if (seen.has(intent.tag))
                throw Error('Please fix the dataset, tag "' + intent.tag + "' is redundant.");
            seen.add(intent.tag);

            this.intent_func[intent.tag] = null; // a function associated with this state
        }
    }

    /**
     * @param {string[]} words_list 
     */
    buildVocabularyUsing (words_list) {
        this.vocabulary = words_list;
    }

    /**
     * @param {Function?} criteria_func 
     */
    buildVocabularyFromDataset (criteria_func = null) {
        if (this.intents.length == 0)
            throw Error ('No intent data loaded');
        const set_words = new Set();
        for (let intent of this.intents) {
            const tokens = ShioriNLP.tokenize(intent.patterns.join(' '));
            for (let token of tokens) {
                if (criteria_func) {
                    const accepted = criteria_func(token);
                    if (accepted) // accept under certain conditions
                        set_words.add(token);
                    continue;
                }
                set_words.add(token);
            }
        }

        this.vocabulary = [...set_words];
    }

    /**
     * @param {string} input_sentence 
     * @returns 
     */
    produceInputVectorFrom (input_sentence) {
        const input_vector = []; // same size as the vocabulary array
        // input vector (according to the vocabulary)
        const tokens = ShioriNLP.tokenize (input_sentence);
        let max = -Infinity;
        for (let word of this.vocabulary) {
            let value = 0;
            if (tokens.includes(word)) {
                value = this.consider_word_order ? (1 + tokens.indexOf(word)) : 1;
                max = Math.max(value, max);
            }
            input_vector.push(value);
        }

        if (this.consider_word_order)
            return input_vector.map(value => value / Math.max(0., max));

        return input_vector;
    }

    /**
     * @private
     * Simple utility function for numerizing the input
     * @param {string} label 
     * @param {string} input_sentence 
     * @returns 
     */
    produceLabeledInputFrom (label, input_sentence) {
        // return patterns.map(ShioriNLP.tokenize)
        const input_vector = this.produceInputVectorFrom(input_sentence);
        const label_vector = []; // same size as the number of tags
        // label vector (supervised output)
        for (let {tag} of this.intents)
            label_vector.push(tag == label ? 1 : 0);

        return {
            label_vector : label_vector,
            input_vector : input_vector
        };
    }

    /**
     * @private
     * @returns [{label_vector: number[], input_vector: number[]}, ..]
     */
    produceNumericDataset () {
        const training_dataset = [];
        for (let {tag, patterns} of this.intents) {
            for (let sentence of patterns) {
                const data = this.produceLabeledInputFrom(tag, sentence);
                training_dataset.push(data);
            }
        }
        return training_dataset;
    }

    /**
     * @param {number} epochs 
     * @param {Function} callback_fun 
     */
    async train (epochs = 500, callback_fun) {
        const dataset = this.produceNumericDataset();
        const model = tf.sequential();
        // note
        // units : number of neurons
        // activation : activation function
        // inputShape : shape of the input (reshape() logic if we use an array with 3 values)
        
        // dense is a fully connected layer
        const n_input = this.vocabulary.length;
        const n_output = this.intents.length;

        // hidden
        model.add (tf.layers.dense({
            units: 100, activation: 'sigmoid', inputShape: [n_input]
        }));

        // output
        model.add (tf.layers.dense({
            units: n_output, activation: 'sigmoid'
        }));
        
        // sgdOpt
        // we can also use adam but let's stick with the old gradient descent for now
        const sgdOpt = tf.train.sgd (0.2); // learning rate
        // model.compile({optimizer: sgdOpt, loss: 'meanSquaredError'});
        // works better for binary-ish values
        // as it tries to minimize the chaos between the expected value and the output
        model.compile ({optimizer: sgdOpt, loss: 'binaryCrossentropy'});
        
        const input = tf.tensor2d (dataset.map(data => data.input_vector));
        const label = tf.tensor2d (dataset.map(data => data.label_vector));
        
        
        for (let i = 0; i < epochs; i++) {
            let res_history = await model.fit (input, label);
            this.model_loss = res_history.history.loss[0];
            if (callback_fun)
                callback_fun(res_history, i + 1);
        }

        input.dispose ();
        label.dispose ();

        this.model = model;

        return this.model;
    }

    /**
     * @param {string} sentence 
     * @returns 
     */
    predict (sentence) {
        if (!this.model)
            throw Error('Please train a model first !');
        const input_vector = this.produceInputVectorFrom (sentence);
        const input_tensor = tf.tensor2d ([input_vector]);
        const prediction = this.model.predict (input_tensor);
        const as_array = Array.from (prediction.dataSync());
        const prediction_result = {};
        for (let {tag} of this.intents)
            prediction_result[tag] = as_array.shift ();
        return prediction_result;
    }

    /**
     * @param {string} str 
     * @returns string[]
     */
    static tokenize (str) {
        return str
                .trim()
                .toLowerCase()
                .replace(/['\-]+/g, '')
                .replace(/[ .?!]+/g, ' ')
                .split(/[ ,;\t\n]+/g);
    }
}