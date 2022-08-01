const { assert } = require('console');
const fs = require ('fs');
const ShioriNLP = require('./ShioriNLP');
const EPSILON = 10e-6;

class WordVector {
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
        for (let i = 0; i < this.components.length; i++) {
            if (Math.abs (wordvec.components[i] - this.components[i]) > EPSILON)
                return false;
        }
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
     * * dot(a, b) = |a||b| * cos(angle(a, b))
     * * costDist := cost(t) 
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static cosDist (a, b) {
        return a.dot(b) / (a.length() * b.length());
    }

    /**
     * @param {WordVector} a 
     * @param {WordVector} b 
     * @returns 
     */
    static dimCheck (a, b) {
        if (a.dim() != b.dim())
            throw Error ('Operands does not share the same dimensionality');
    }

    toString () {
        return `WordVector [${this.components.join(', ')}]`;
    }
}

class ShioriWord2Vec {
    
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
        const tokens = ShioriNLP.tokenize (this.text.toLowerCase());
        
        const place_holder_expr = '__';
        const isValidIndex = x => x >= 0 && x < tokens.length;
        const reconstrOriginal = (hash, token) => hash.replace(place_holder_expr, token);

        let cursor = 0;
        const column_set = new Set();
        const occ_count = {};
        while (cursor < tokens.length) {
            let current = [];
            for (let i = (cursor - n_context); i <  (cursor + n_context + 1); i++) {
                if (!isValidIndex(i)) 
                    continue;
                current.push(i == cursor ? place_holder_expr : tokens[i]);
            }
            const hash_key = current.join(' ');
            const original = reconstrOriginal (hash_key, tokens[cursor]); // He _ angry => He was angry
            occ_count[original] = occ_count[original] ? (occ_count[original] + 1) : 1;

            column_set.add (hash_key);
            cursor++;
        }
        // console.log(column_set.size, column_set);
        // console.log(occ_count)
        
        // building the vector
        const words = {};
        for (let token of tokens) {
            if (words[token]) // seen
                continue;
            words[token] = new WordVector(); // init the vector
            for (let col of column_set) { // each column == a single component in this approach
                const original = reconstrOriginal (col, token);
                const component = occ_count[original] || 0;
                words[token].addComponent (component); 
            }
        }

        return words;
    }


    word2vec (str) {
        if (!this.is_trained_model)
            throw Error ('Please train first');
    }
}


module.exports = {
    ShioriWord2Vec : ShioriWord2Vec,
    WordVector : WordVector
};