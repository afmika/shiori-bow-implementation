const ShioriNLP = require('../core/ShioriNLP');
const nlp = new ShioriNLP();

nlp.load('../datas/basic-intents.json');

const customFilter = (token) => {
    return token.length > 1;
};

nlp.buildVocabularyFromDataset(customFilter);

// console.log(nlp.vocabulary)
// we should farm on the same dataset
console.log(nlp.produceLabeledInputFrom('greeting', 'Is anyone there ?'));
console.log(nlp.produceNumericDataset());