const Mat = require ('../core/Mat');

const m = new Mat(
    [1, 2, 3],
    [4, -1, 6]
);

const b = new Mat(
    [1, 2, 3], 
    [4, -1, 6], 
    [5, 8, 9]
);

const c = new Mat(
    [0, 0.5, 4], 
    [-4, -1, 6], 
    [0, -1, 3]
);

console.log(m.dim());
console.log(m.prod(b));
console.log(c.add(b).scale(.5));
console.log(Mat.Identity(4).map((v, i, j) => v * (i + j + 1)));
console.log(Mat.covec(1, 1, 1));
console.log(Mat.vec(1, 1, 1));
console.log('trivial prod', Mat.vec(1, 2, 3).prod(Mat.covec(-1, 0, 1)));
console.log('trivial dot', Mat.covec(-2, 0, 1).prod(Mat.vec(1, 1, 1)));