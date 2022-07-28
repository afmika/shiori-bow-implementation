module.exports = class ShioriNLP {
    constructor () {
        this.intents = [];
        this.intent_func = {};

        this.vocabulary = [];
    }

    /**
     * @param {string} filename 
     */
    load (filename) {
        const intents_array = require(filename);
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

    train () {
        if (this.intents.length == 0)
            throw Error ('No intent data loaded');
        
    }

    /**
     * @private
     * Simple utility function for numerizing the input
     * @param {string} label 
     * @param {string} input_sentence 
     * @returns 
     */
    produceInputFrom (label, input_sentence) {
        // return patterns.map(ShioriNLP.tokenize)
        const label_vector = []; // same size as the number of tags
        const input_vector = []; // same size as the vocabulary array

        // label vector (supervised output)
        for (let {tag} of this.intents) {
            label_vector.push(tag == label ? 1 : 0);
        }

        // input vector (according to the vocabulary)
        const tokens = ShioriNLP.tokenize (input_sentence);
        for (let word of this.vocabulary) {
            input_vector.push(tokens.includes(word) ? 1 : 0);
        }

        return {
            label_vector : label_vector,
            input_vector : input_vector
        };
    }

    produceNumericDataset () {
        const training_dataset = [];
        for (let {tag, patterns} of this.intents) {
            for (let sentence of patterns) {
                const data = this.produceInputFrom(tag, sentence);
                training_dataset.push(data);
            }
        }
        return training_dataset;
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