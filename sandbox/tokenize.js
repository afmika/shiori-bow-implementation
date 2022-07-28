const ShioriNLP = require('../core/ShioriNLP');
const nlp = new ShioriNLP();
const tokens = ShioriNLP.tokenize("Hello world, doesn't it look cool   !? chuck-nuts");
console.log(tokens);