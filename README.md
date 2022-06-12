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

Then start a webserver. I recommend the rust devserver

    cargo install devserver
    devserver

Finally, load [the page](http://localhost:8080/) (this link is using port 8080 which is the default devserver port).

## author
vidstige
