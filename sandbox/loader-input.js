const ShioriNLP = require('../core/ShioriNLP');
const nlp = new ShioriNLP();

nlp.load('./datas/basic-intents.json');

const customFilter = (token) => {
    return token.length > 1;
};

nlp.buildVocabularyFromDataset(customFilter);

// console.log(nlp.vocabulary)
// we should farm on the same dataset
nlp.consider_word_order = true;
console.log(nlp.produceInputVectorFrom('Is anyone there here ?'));
console.log(nlp.produceInputVectorFrom('Is there anyone here ?'));
// console.log(nlp.produceLabeledInputFrom('greeting', 'Is anyone there ?'));
// console.log(nlp.produceNumericDataset());