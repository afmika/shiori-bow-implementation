const ShioriNLP = require('../core/ShioriNLP');
const {ShioriWord2Vec, WordVector} = require('../core/ShioriWord2Vec');

const sw2v = new ShioriWord2Vec ();
sw2v.loadVectorsFromFile ('./test-output.json')
console.log('Data infos', sw2v.infos());

const result = WordVector.add (
    sw2v.word2vec ('Mother'),
    sw2v.word2vec ('father')
);
const top = sw2v.closestWordByVector (result, 10);
console.log(top.map(v => v)); // should be the 'girl' ;)