const ShioriBOT = require("./core/ShioriBOT");

const bot = new ShioriBOT();
const consider_word_order = false;
bot.setup('./datas/basic-intents.json', consider_word_order)
    .then(nlp => {
        console.log('Done training');
        console.log(bot.respond('Hello world'));
    })
    .catch(err => {
        console.error(err);
    });