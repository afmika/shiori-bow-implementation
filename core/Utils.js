module.exports = class Utils {
    static safeRun (fun) {
        if (fun && typeof fun !== 'function')
            return null;
        return fun || ((...arg) => {
            return null;
        });
    }
};