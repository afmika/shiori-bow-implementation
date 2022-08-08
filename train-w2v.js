const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

sw2v.loadTextFromFile (
    './datas/konosuba.en.txt',
    './datas/exclude.txt'
);

sw2v.max_vec_dimension = 100;
const window_size = 2;
(async () => {
    await sw2v.trainTensorFlow (window_size, 100, (loss, n_current, total) => {
        const p = Math.floor (100 * n_current / total);
        console.log ('Training : '+ p + '% :: loss ', loss, n_current + '/' + total);
    });

    sw2v.saveVectorsTo ('./trained/shiori-w2v/konosuba.en-vec.json');
}) ();