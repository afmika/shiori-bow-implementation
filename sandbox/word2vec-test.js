const ShioriNLP = require('../core/ShioriNLP');
const {ShioriWord2Vec, WordVector} = require('../core/ShioriWord2Vec');

// const input = 'It was the best of times, it was the worst of times.';
// const sw2v = new ShioriWord2Vec (input);

// const vectors = sw2v.train (1);
// console.log (vectors);

// const a = new WordVector ([1, 1]);
// const b = new WordVector ([1, 0]);
// console.log(WordVector.cosDist(a, b));


// const sw2v = new ShioriWord2Vec ();
// sw2v.loadTextFromFile ('./datas/gutenberg-books/mix.txt');
// sw2v.train (1);
// // console.log(sw2v.words)
// console.log(sw2v.closestWord('she', 5)); // should be the 'girl' ;)


const sw2v = new ShioriWord2Vec ();
sw2v.loadTextFromFile ('./datas/gutenberg-books/mix.txt');
// sw2v.loadTextFromFile ('./datas/konosuba.en.txt');
const exclude = ['the', 'a', 'an'];
// sw2v.max_vec_dimension = 1000;
sw2v.train (1, exclude);

const result = WordVector.add (
    sw2v.word2vec ('Mother'),
    sw2v.word2vec ('girl')
);

console.log('Total words', sw2v.infos());

const top = sw2v.closestWordByVector (result, 10);
console.log(top.map(v => v.word)); // should be the 'girl' ;)