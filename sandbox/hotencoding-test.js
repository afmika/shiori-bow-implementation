const ShioriNLP = require('../core/ShioriNLP');
const {ShioriWord2Vec, WordVector} = require('../core/ShioriWord2Vec');

const sw2v = new ShioriWord2Vec ();
sw2v.loadTextFromFile (
    './datas/sample.txt'
);

console.log(sw2v.hotEncode('natural'));
console.log(sw2v.hotEncode('processing'));
console.log(sw2v.hotEncode('exciting'));

console.log(sw2v.generateTrainingDatas(2).pop())