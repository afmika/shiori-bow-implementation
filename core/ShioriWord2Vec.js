const fs = require ('fs');
const ShioriNLP = require('./ShioriNLP');

module.exports = class ShioriWord2Vec {
    
    /**
     * @param {string[]} vocabulary 
     */
    constructor (text) {
        this.text = text || '';
        this.is_trained_model = false;
    }

    /**
     * @param {string} filename
     */
    loadTextFromFile (filename) {
        this.text = fs
                    .readFileSync (filename)
                    .toString();
    }

    /**
     * @param {number} n_context number of 'context' word on the left/right of a given token
     */
    train (n_context = 1) {
        const tokens = ShioriNLP.tokenize (this.text);
        
        let cursor = 0;
        const isValidIndex = x => x >= 0 && x < tokens.length;

        const column_set = new Set();
        while (cursor < tokens.length) {
            let current = [];
            for (let i = (cursor - n_context); i <  (cursor + n_context + 1); i++) {
                if (!isValidIndex(i)) 
                    continue;
                if (i != cursor) {
                    current.push(tokens[i]);
                } else {
                    current.push('__');
                }
            }
            const hash_key = current.join(' ');
            column_set.add(hash_key);
            cursor++;
        }

        // console.log(column_set.size, column_set);
    }


    word2vec (str) {
        if (!this.is_trained_model)
            throw Error ('Please train first');
    }
}