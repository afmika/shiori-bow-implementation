const ShioriNLP = require('../core/ShioriNLP');
const nlp = new ShioriNLP();

// load the dataset
nlp.load('../datas/basic-intents.json');

// build a vocabulary from it
nlp.buildVocabularyFromDataset((token) => {
    return token.length > 1; // remove words with length <= 1
});

nlp
    .train(100, (res, epoch) => {
        console.log(epoch, res.history.loss[0])
    })
    .then(async (model) => {
        console.log(nlp.predict('What is your name ?'));
    });

// console.log(nlp.produceNumericDataset());