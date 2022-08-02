module.exports = class Utils {
    static safeRun (fun) {
        return fun || ((...arg) => {
            return null;
        });
    }
};