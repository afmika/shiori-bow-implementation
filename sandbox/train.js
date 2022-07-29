const ShioriNLP = require('../core/ShioriNLP');
const nlp = new ShioriNLP();

// load the dataset
nlp.load('../datas/basic-intents.json');

// build a vocabulary from it
nlp.buildVocabularyFromDataset((token) => {
    return token.length > 1; // remove words with length <= 1
});

nlp
    .train(500, (res, epoch) => {
        console.log(res.history.loss[0])
    })
    .then(() => {
        console.log(nlp.predict('Hello'));
    });

// console.log(nlp.produceNumericDataset());