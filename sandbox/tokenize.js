const ShioriNLP = require('../core/ShioriNLP');
const nlp = new ShioriNLP();
const tokens = ShioriNLP.tokenize("Hello world, doesn't it look cool   !? chuck-nuts 1234 28.65qqq");
console.log(tokens);