const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();
sw2v.loadTextFromFile (
    './datas/gutenberg-books/pride-and-prejudice.txt',
    './datas/exclude.txt'
);
// sw2v.loadTextFromFile ('./datas/konosuba.en.txt');
sw2v.max_vec_dimension = 2000;
let current_mult = 0, state = null;
sw2v.trainOptimally (1, (msg, n_current, total) => {
    const p = Math.floor (100 * n_current / total);
    const k = Math.floor (p / 20);
    const r = p - k * 20;
    const curr_state = msg.split(':')[0];
    if (curr_state != state) {
        state = curr_state;
        current_mult = 0;
    }
    if (k > current_mult && r == 0) {
        console.log ('Training : '+ p + '%', 'status ' + msg, n_current + '/' + total);
        current_mult = k;
    }
});

const result = WordVector.add (
    sw2v.word2vec ('Mother'),
    sw2v.word2vec ('girl')
);

console.log('Total words', sw2v.infos());

const top = sw2v.closestWordByVector (result, 10);
console.log(top.map(v => v.word)); // should be the 'girl' ;)

sw2v.closestWordByVector (WordVector.add (
    sw2v.word2vec ('girl'),
    sw2v.word2vec ('expression'),
), 5);