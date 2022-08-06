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

console.log('\n# basics');
m.prod(b).print();
c.add(b).scale(.5).print();
Mat.Identity(4).map((v, i, j) => v * (i + j + 1)).print();

console.log('\n# vectors');
Mat.covec(1, 1, 1).print();
Mat.vec(1, 1, 1).print();
(Mat.vec(1, 2, 3).prod(Mat.covec(-1, 0, 1))).print();
(Mat.covec(-2, 0, 1).prod(Mat.vec(1, 1, 1))).print();

console.log('\n# random');
// random matrix Mij = rand(-1, 1)
Mat.rand(3, 2, x => 2 * x - 1).print();

console.log('\n# misc');
const aa = new Mat([1, 2, 3],[4, 5, 6], [7, 8, 9]);
const bb = new Mat([1, 0],[0, 1]);;
// aa.tensorProd(bb).print();
aa.tensorProd(bb).print();
bb.tensorProd(aa).print();

console.log('\n# outer product');
const u = new Mat([1,2], [3, 4]);
const v = new Mat([1,2,3], [4,0,3]);

u.outerProd(v).print();