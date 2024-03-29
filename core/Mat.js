module.exports = class Mat {
    /**
     * @param  {...number[]} entries 
     */
    constructor (...entries) {
        if (entries.length == 0) 
            return;
        if (entries.length == 1) {
            // single arg ==> an array[][]
            // the most optimised input
            this.entries = entries[0];
            return;
        }
        if (typeof entries[0] == 'number') {
            this.entries = [entries];
            return;
        }
        // list of rows!
        this.entries = new Array (...entries);
    }

    /**
     * Produce a row vector
     * @param  {...number} entries 
     */
    static covec (...entries) {
        return new Mat([entries]);
    }

    /**
     * Produce a column vector
     * @param  {...number} entries 
     */
    static vec (...entries) {
        return (new Mat([entries])).transpose();
    }

    /**
     * @param {*} n_row 
     * @param {*} n_col 
     * @param {*} fn 
     */
    static rand (n_row, n_col, fn) {
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(n_col);
            for (let j = 0; j < n_col; j++) {
                const rn = Math.random ();
                result[i][j] = fn ? fn(rn, i, j) : rn;
            }
        }
        return new Mat(result);
    }

    /**
     * @param {number} n 
     */
    static Identity (n = 2) {
        let result = new Array(n);
        for (let i = 0; i < n; i++) {
            result[i] = new Array(n);
            for (let j = 0; j < n; j++)
                result[i][j] = i == j ? 1 : 0;
        }
        return new Mat(result);
    } 

    /**
     * @param {Mat} other 
     * @returns {boolean}
     */
    dimCompare (other) {
        const a = this.dim();
        const b = other.dim();
        return a.n_col == b.n_col && a.n_row == b.n_row;
    }

    /**
     * @returns {boolean}
     */
    isSquare () {
        const {n_row, n_col} = this.dim();
        return n_row == n_col;
    }

    /**
     * @param {number} i 
     * @param {number} j 
     * @returns {number}
     */
    get (i, j) {
        return this.entries[i][j];
    }

    /**
     * @param {number} index 
     */
    getSingleIndexed (index) {
        const {n_col} = this.dim();
        const j = index % n_col;
        const i = Math.floor(index / n_col);
        return this.get (i, j);
    }

    /**
     * @param {Function} fun Function(value, i, j) 
     * @returns {Mat}
     */
    map (fun) {
        const {n_row, n_col} = this.dim();
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(n_col);
            for (let j = 0; j < n_col; j++)
                result[i][j] = fun(this.get(i, j), i, j);
        }
        return new Mat(result);
    }

    /**
     * @param {Function} fun Function(value, i, j)
     */
    each (fun) {
        const {n_row, n_col} = this.dim();
        for (let i = 0; i < n_row; i++) {
            for (let j = 0; j < n_col; j++)
                fun(this.get(i, j), i, j);
        }
    }

    /**
     * @returns \{n_row, n_col\}
     */
    dim () {
        return {
            n_row : this.entries.length,
            n_col : this.entries[0].length
        };
    }

    /**
     * @returns {Mat}
     */
    transpose () {
        let {n_row, n_col} = this.dim();
        // now swap
        [n_row, n_col] = [n_col, n_row];
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(n_col);
            for (let j = 0; j < n_col; j++)
                result[i][j] = this.get(j, i);
        }
        return new Mat(result);
    }

    /**
     * @param {Mat} b 
     * @returns 
     */
    prod (b) {
        const {n_row, n_col} = this.dim();
        if (n_col != b.dim().n_row)
            throw Error ('first operand n_row != second operand n_col');
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(b.dim().n_col);
            for (let j = 0; j < b.dim().n_col; j++) {
                let s_dot = 0;
                for (let k = 0; k < n_col; k++) 
                    s_dot += this.get(i, k) * b.get(k, j);
                result[i][j] = s_dot;
            }
        }
        return new Mat(result);
    }

    /**
     * Computes a^T b without calling a.transpose()
     * @param {Mat} a 
     * @param {Mat} b 
     * @returns 
     */
    static prodTransposeLeft (a, b) {
        let {n_row, n_col} = a.dim();
        // swap
        [n_row, n_col] = [n_col, n_row];

        if (n_col != b.dim().n_row)
            throw Error ('first operand n_row != second operand n_col');
        
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(b.dim().n_col);
            for (let j = 0; j < b.dim().n_col; j++) {
                let s_dot = 0;
                for (let k = 0; k < n_col; k++) 
                    s_dot += a.get(k, i) * b.get(k, j); // just swap a's indices
                result[i][j] = s_dot;
            }
        }
        return new Mat(result);
    }

    /**
     * @param {Mat} b 
     * @returns {Mat}
     */
    tensorProd (b) {
        const a_dim = this.dim();
        const b_dim = b.dim();
        // (N x M) (x) (N' x M')
        let total_row = a_dim.n_row * b_dim.n_row;
        let total_col = a_dim.n_col * b_dim.n_col;
        const result = new Array ();
        for (let i = 0; i < total_row; i++)
            result[i] = new Array(total_col);
        
        for (let i = 0; i < a_dim.n_row; i++) {
            for (let j = 0; j < a_dim.n_col; j++) {
                const aij = this.get(i, j);
                const by = i * b_dim.n_row, bx = j * b_dim.n_col; 
                b.each ((bkl, r, c) => {
                    result[by + r][bx + c] = aij * bkl;
                });
            }
        }

        return new Mat(result);
    }
    /**
     * * a := flatten current matrix
     * * b := flatten matrix b
     * * Out ij = a[i]*b[j]
     * * Out row i = a[i]*b
     * @param {Mat} b
     * @returns {Mat} (N x M) x (N' x M') 
     */
    outerProd (b) {
        const a_dim = this.dim();
        const b_dim = b.dim();
        // (N x M) (x) (N' x M')
        let total_row = a_dim.n_row * a_dim.n_col;
        let total_col = b_dim.n_row * b_dim.n_col;
        const result = new Array(total_row);
        for (let i = 0; i < total_row; i++) {
            result[i] = new Array(total_col);
            for (let j = 0; j < total_col; j++)
                result[i][j] = this.getSingleIndexed(i) * b.getSingleIndexed(j);
        }
        
        return new Mat(result);
    }

    /**
     * @param {Mat} b 
     * @returns {Mat}
     */
    add (b) {
        if (!this.dimCompare(b))
            throw Error ('Not of the same dimnension!');
        const {n_row, n_col} = this.dim();
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(n_col);
            for (let j = 0; j < n_col; j++)
                result[i][j] = this.get(i, j) + b.get(i, j);
        }
        return new Mat(result);
    }

    /**
     * @param {Mat} b 
     * @returns {Mat}
     */
    sub (b) {
        if (!this.dimCompare(b))
            throw Error ('Not of the same dimnension!');
        const {n_row, n_col} = this.dim();
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(n_col);
            for (let j = 0; j < n_col; j++)
                result[i][j] = this.get(i, j) - b.get(i, j);
        }
        return new Mat(result);
    }

    /**
     * @param {number} k
     * @returns {Mat}
     */
    scale (k) {
        const {n_row, n_col} = this.dim();
        let result = new Array(n_row);
        for (let i = 0; i < n_row; i++) {
            result[i] = new Array(n_col);
            for (let j = 0; j < n_col; j++)
                result[i][j] = this.get(i, j) * k;
        }
        return new Mat(result);
    }

    /**
     * @param {Function} fn Function (accumulator, value)
     * @param {number} start Ex: pick 0 for an additive reduction, 1 for a multiplicative reduction
     * @returns {number}
     */
    foldToScalar (fn, start = 0) {
        let result = start || 0;
        this.each((v, i, j) => {
            result = fn(result, v)
        });
        return result;
    }

    /**
     * Shorthand for a more verbose console.log
     */
    print (disable) {
        const {n_row, n_col} = this.dim();
        let str = `Mat ${n_row}x${n_col}\n`
            + '[ '
            + this.entries
                    .map((row, i) => (i > 0 ? '  ' : '') + row.join(' '))
                    .join('\n')
            + ' ]';
        if (!disable) console.log(str);
        return str;
    }

    
    /**
     * Print the shape of the current  matrix
     */
     printShape (disable) {
        const {n_row, n_col} = this.dim();
        let str = `Mat ${n_row}x${n_col}\n`;
        if (!disable) console.log(str);
        return str;
    }
}