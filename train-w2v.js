const {ShioriWord2Vec, WordVector} = require('./core/ShioriWord2Vec');
const sw2v = new ShioriWord2Vec ();

sw2v.loadTextFromFile (
    './datas/konosuba.en.txt',
    './datas/exclude.txt'
);

sw2v.max_vec_dimension = 20;
const window_size = 2;
const n_epochs = 500;
(async () => {
    await sw2v.trainOptimally (window_size, n_epochs, (loss, delta_loss, n_current, total) => {
        const p = Math.floor (100 * n_current / total);
        const sign = delta_loss < 0 ? '[ UP ]' : '';
        console.log ('Training : '+ p + '% :: loss ',  loss + sign,':: delta_loss ', delta_loss, n_current + '/' + total);
    });

    sw2v.saveVectorsTo ('./trained/shiori-w2v/konosuba.en-vec.json');
}) ();