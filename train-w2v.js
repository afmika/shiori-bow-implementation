const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

sw2v.loadTextFromFile (
    './datas/sample.txt',
    './datas/exclude.txt'
);

sw2v.max_vec_dimension = 300;
const window_size = 1 ;
(async () => {
    await sw2v.trainOptimally (window_size, 5, (loss, n_current, total) => {
        const p = Math.floor (100 * n_current / total);
        console.log ('Training : '+ p + '% :: loss ', loss, n_current + '/' + total);
    });

    sw2v.saveVectorsTo ('./trained/shiori-w2v/mix-vec.json');
}) ();