const ShioriNLP = require('../core/ShioriNLP');
const ShioriWord2Vec = require('../core/ShioriWord2Vec');

const input = 'It was the best of times, it was the worst of times.';
const sw2v = new ShioriWord2Vec (input);

// sw2v.loadVocabularyFromFile ('./datas/vocab.nips.txt');
console.log(sw2v.train(1));
// sw2v.setup(3);