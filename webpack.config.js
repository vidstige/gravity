module.exports = {
    entry: [
        "./src/index.js",
        "./static/index.html",
    ],
    output: {
        path: __dirname + "/build",
        filename: "bundle.js",
    },
    module: {
        loaders: [
            { test: /\.(obj|png|ttf|svg)$/, loader: 'url-loader?limit=100000' },
            { test: /\.tff$/, loader: 'file-loader?name=fonts/[name].[ext]' },
            { test: /\.css$/, loader: "style-loader!css-loader" },
            { test: /\.html/,  loader: 'file-loader?name=[name].[ext]' },
        ]
    },
    devServer: {
        contentBase: "static/",
    }
};
