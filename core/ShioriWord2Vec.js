const fs = require ('fs');
const ShioriNLP = require('./ShioriNLP');
const Utils = require('./Utils');
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
        this.excluded_tokens = [];
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
            this.tokens = this.tokens.filter(it => !this.excluded_tokens.includes(it))
        }
    }

    /**
     * * The number of columns will be equal to min(computed_col_length, max_vec_dimension)
     * @param {number?} n_context number of 'context' word on the left/right of a given token
     * @param {Function?} log_fun Function (message, n_current, total)
     */
    train (n_context = 1, log_fun = null) {
        let tokens = this.tokens;

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
            if (column_set.size >= this.max_vec_dimension)
                break; // force 
            cursor++;
            Utils.safeRun (log_fun) ('1:construct_column', cursor, tokens.length);
        }
        // console.log(column_set.size, column_set);
        // console.log(occ_count)
        
        // building the vector
        const words = {};
        let pos = 1;
        for (let token of tokens) {
            Utils.safeRun (log_fun) ('2:construct_word_vector', pos++, tokens.length);
            if (words[token]) // seen
                continue;
            words[token] = new WordVector(); // init the vector
            for (let col of column_set) { // each column == a single component in this approach
                const original = reconstrOriginal (col, token);
                const component = occ_count[original] || 0;
                words[token].addComponent (component); 
            }
        }

        this.is_trained_model = true;
        this.words = words;
    }


    /**
     * * Tries its best to reduce the number of vector columns while retaining the relevant ones
     * * The number of columns will be equal to min(computed_col_length, max_vec_dimension)
     * @param {number?} n_context number of 'context' word on the left/right of a given token
     * @param {Function?} log_fun Function (message, n_current, total)
     */
     trainOptimally (n_context = 1, log_fun = null) {
        let tokens = this.tokens;

        const place_holder_expr = '__';
        const isValidIndex = x => x >= 0 && x < tokens.length;
        const reconstrOriginal = (hash, token) => hash.replace(place_holder_expr, token);

        let cursor = 0;
        const column_set = new Set();
        const occ_count = {};
        const share_count = {};
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
            share_count[hash_key] = share_count[hash_key] ? (share_count[hash_key] + 1) : 1;

            column_set.add (hash_key);
            cursor++;
            Utils.safeRun (log_fun) ('1:construct_column', cursor, tokens.length);
        }
        // console.log(column_set.size, column_set);
        // console.log(occ_count)

        // the main idea here is to reduce the column set
        const sorted_column_array = [...column_set];
        // in this approach, we define the most shared column as more 'relevant'
        // my initial approach was to use the less used ones as they are more likely to be a feature
        // for a given word (carry more information so to speak)
        // but it seems not working very well with vector arithmetic

        // .. then I realized that the most 'shared' column here means columns that are more likely to give
        // context to all words, it's the 'word' (token) that carries information not the surroundings !
        // for a given word (information) we need to assign it with a 'context' i.e. we give it a meaning !
        // Ex: his name was {x}, he was hit by a truck
        // This sentence clearly depends on {x} i.e. x gives a meaning/purpose to its surrounding context
        // by ordering them while prioritizing the most shared ones we can
        //
        // Ex: this kid is actually a {x}
        // {x} = boy, girl, power-ranger
        // => boy, girl and should should have cosine_distance ~ 1
        // => the vector produced are not equal (different length) but oriented in the same way
        sorted_column_array.sort((b, a) => share_count[a] - share_count[b]);
        
        // building the vector
        const words = {};
        let pos = 1;
        for (let token of tokens) {
            Utils.safeRun (log_fun) ('2:construct_word_vector', pos++, tokens.length);
            if (words[token]) // seen
                continue;
            words[token] = new WordVector (); // init the vector
            const max_length = Math.min (this.max_vec_dimension, sorted_column_array.length);
            for (let i = 0; i < max_length; i++) { // each column == a single component in this approach
                const col = sorted_column_array [i];
                const original = reconstrOriginal (col, token);
                const component = occ_count[original] || 0;
                words[token].addComponent (component); 
            }
        }

        this.is_trained_model = true;
        this.words = words;
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
            top_list.push({word : item, dist : dist, cosine_dist : cosine_dist, vec : vec});
        }

        top_list
            .sort((a, b) => {
                let dist_diff = a.dist - b.dist;
                // we do not care about the orientation, we just care
                // about the angle between the two vectors
                // the bigger the cosine diff, the more similar the vectors are !
                let cos_diff = Math.abs(b.cosine_dist) - Math.abs(a.cosine_dist);
                if (isNaN (cos_diff)) // ex +Infinity-Infinity which is undefined
                    cos_diff = 0;
                // prioritize the cosine dist
                return Math.abs(cos_diff) < EPSILON ? dist_diff : cos_diff;
            });

        return top_list.filter((_, i) => i < top);
    }

    /**
     * @returns infos
     */
    infos () {
        const vec_dim = this.words [ Object.keys(this.words) [0] ].dim();
        return {total_words : Object.keys(this.words).length, vec_dim : vec_dim};
    }
}


module.exports = {
    ShioriWord2Vec : ShioriWord2Vec,
    WordVector : WordVector
};