const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

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

// function scale (w1, factor) {
//     const v = WordVector.scale(vec (w1), factor);
// }

function mean (w1, w2, top = 3) {
    return sw2v.meanWord (w1, w2, top);
}

sw2v.loadVectorsFromFile ('./trained/shiori-w2v/test-vec.json');
console.log('Infos :: ', sw2v.infos());