const Mat = require("../core/Mat");

// 2 x 5
let a = new Mat(
    [0, 1, 0, 1, 4],
    [-1, 0, 1, 5, 6]
);

// 2 x 1
let b = Mat.vec(1, 2);

const target = a.transpose().prod(b);
const res = Mat.prodTransposeLeft(a, b);
target.print();
res.print();