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
     */
    async setup (filename) {
        const nlp = new ShioriNLP();
        this.nlp = nlp;

        nlp.load(filename);
        nlp.buildVocabularyFromDataset();
        
        await nlp.train(this.default_epoch);

        console.info('Error loss value', this.nlp.model_loss);
    }

    /**
     * @param {string} input_sentence 
     */
    respond (input_sentence) {
        const result = this.nlp.predict(input_sentence);
        const ordered = [];
        
        for (let tag in result)
            ordered.push({tag : tag, score : result[tag]});
        
        // descending order
        ordered.sort((a, b) => b.score - a.score);
        console.log(ordered);
        
        // tag attribute of the first item first item
        const [{tag}, ] = ordered; 
        const [{responses}, ] = this.nlp.intents.filter(intent => intent.tag == tag);
        return responses [Math.floor(Math.random() * responses.length)];
    }
}