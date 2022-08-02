const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

sw2v.loadVectorsFromFile ('./trained/shiori-w2v/shakespear-vectors.json');
// sw2v.loadTextFromFile (
//     './datas/gutenberg-books/t8.shakespeare.txt',
//     './datas/exclude.txt'
// );
// sw2v.max_vec_dimension = 2500;
// let current_mult = 0, state = null;
// sw2v.trainOptimally (1, (msg, n_current, total) => {
//     const p = Math.floor (100 * n_current / total);
//     const chunk_size = 5;
//     const k = Math.floor (p / chunk_size);
//     const r = p - k * chunk_size;
//     const curr_state = msg.split(':')[0];
//     if (curr_state != state) {
//         state = curr_state;
//         current_mult = 0;
//     }
//     if (k > current_mult && r == 0) {
//         console.log ('Training : '+ p + '%', 'status ' + msg, n_current + '/' + total);
//         current_mult = k;
//     }
// });

// sw2v.saveVectorsTo ('./shakespear-vectors.json')

console.log('Infos :: ', sw2v.infos());

function vec (w) {
    return sw2v.word2vec (w);
}

function closest (w, top = 3) {
    return sw2v.closestWord (w, top);
}

function closestVec (v, top = 3) {
    return sw2v.closestWordByVector (v, top);
}

function add(w1, w2, top = 3) {
    const vec = WordVector.add(sw2v.word2vec (w1), sw2v.word2vec (w2));
    return sw2v.closestWordByVector (vec, top);
}

function sub(w1, w2, top = 3) {
    const vec = WordVector.sub(sw2v.word2vec (w1), sw2v.word2vec (w2));
    return sw2v.closestWordByVector (vec, top);
}