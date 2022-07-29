const ShioriBOT = require("./core/ShioriBOT");

const bot = new ShioriBOT();

bot.setup('./datas/basic-intents.json')
    .then(nlp => {
        console.log('Done training');
        console.log(bot.respond('Hello world'));
    })
    .catch(err => {
        console.error(err);
    });