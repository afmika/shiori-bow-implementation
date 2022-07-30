const ShioriNLP = require("./ShioriNLP");

module.exports = class ShioriBOT {

    /**
     * @param {ShioriNLP} nlp_engine 
     */
    constructor (nlp_engine) {
        this.nlp = nlp_engine || null;
        this.default_epoch = 2000;
    }

    /**
     * @param {string} filename 
     * @param {boolean?} consider_word_order 
     */
    async setup (filename, consider_word_order = false) {
        const nlp = this.nlp || new ShioriNLP();
        this.nlp = nlp;
        this.nlp.consider_word_order = consider_word_order || false;

        nlp.load (filename);
        nlp.buildVocabularyFromDataset ();
        
        await nlp.train (this.default_epoch);

        console.info ('Error loss value', this.nlp.model_loss);
    }

    /**
     * @param {string} input_sentence 
     */
    respond (input_sentence) {
        const result = this.nlp.predict (input_sentence);
        const ordered = [];
        
        for (let tag in result)
            ordered.push ({tag : tag, score : result[tag]});
        
        // descending order
        ordered.sort ((a, b) => b.score - a.score);
        console.log (ordered);
        
        // tag attribute of the first item
        const [{tag}, ] = ordered; 
        const [{responses}, ] = this.nlp.intents.filter (intent => intent.tag == tag);
        return responses [Math.floor(Math.random() * responses.length)];
    }
}