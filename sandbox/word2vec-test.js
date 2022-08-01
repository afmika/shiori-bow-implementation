const ShioriNLP = require('../core/ShioriNLP');
const {ShioriWord2Vec, WordVector} = require('../core/ShioriWord2Vec');

const input = 'It was the best of times, it was the worst of times.';
const sw2v = new ShioriWord2Vec (input);

// sw2v.loadVocabularyFromFile ('./datas/vocab.nips.txt');
const vectors = sw2v.train (1);
console.log (vectors);
// sw2v.setup(3);

const a = new WordVector ([1, 1]);
const b = new WordVector ([1, 0]);
console.log(WordVector.cosDist(a, b));