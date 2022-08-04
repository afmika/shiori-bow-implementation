const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

sw2v.loadTextFromFile (
    './datas/gutenberg-books/t8.shakespeare.txt',
    './datas/exclude.txt'
);

sw2v.max_vec_dimension = 2000;
let current_mult = 0, state = null;
const window_size = 1 ;
sw2v.trainOptimally (window_size, (msg, n_current, total) => {
    const p = Math.floor (100 * n_current / total);
    const chunk_size = 5;
    const k = Math.floor (p / chunk_size);
    const r = p - k * chunk_size;
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

sw2v.saveVectorsTo ('./trained/shiori-w2v/t8.shakespeare-vector.json');
console.log(sw2v.infos());