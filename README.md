# gravity
Graivty simulation in rust, using Barnes-Hut tree optimization and parallel computing using rayon.

## examples

![gravity simulation](gifs/7.gif)

## running
Running the stand-alone binary is easiest. The gravity program will print raw video to stdout. You can use
ffplay (from the ffmpeg package) to conveniently display it.

    cargo run | ./stream.sh

### wasm
You can also run it in the browser

First build with the web target

    wasm-pack build --target web

Then build and run the web app

    cd ui/
    npm install && npm start

Finally, load [the page](http://localhost:1234/)

## author
vidstige
