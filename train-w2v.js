const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

sw2v.loadTextFromFile (
    './datas/sample.txt',
    './datas/exclude.txt'
);

sw2v.max_vec_dimension = 50;
sw2v.learning_rate = 0.01;

const window_size = 2;
const n_epochs = 3000;
(async () => {
    await sw2v.trainOptimally (window_size, n_epochs, (loss, delta_loss, n_current, total) => {
        const p = Math.floor (100 * n_current / total);
        const sign = delta_loss < 0 ? '*** [ GOES UP ] ***' : '';
        console.log ('Training : '+ p + '% :: loss ',  loss,':: delta_loss ', delta_loss, n_current + '/' + total, sign);
    });

    sw2v.saveVectorsTo ('./trained/shiori-w2v/sample-vec.json');
}) ();