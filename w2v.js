const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();
sw2v.loadTextFromFile ('./datas/gutenberg-books/pride-and-prejudice.txt');
// sw2v.loadTextFromFile ('./datas/konosuba.en.txt');
const exclude = ['the', 'a', 'an'];
sw2v.max_vec_dimension = 2000;
sw2v.train (1, exclude);

const result = WordVector.add (
    sw2v.word2vec ('Mother'),
    sw2v.word2vec ('girl')
);

console.log('Total words', sw2v.infos());

const top = sw2v.closestWordByVector (result, 10);
console.log(top.map(v => v.word)); // should be the 'girl' ;)