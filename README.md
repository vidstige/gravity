# gravity
Graivty simulation in rust, using Barnes-Hut tree optimization and parallel computing using rayon.

## Try online
https://vidstige.github.io/gravity/

## examples

![gravity simulation](gifs/7.gif)

## running
The gravity program will print raw video to stdout. You can use ffplay (from the ffmpeg package) to conveniently display it.

    cargo run --release

### wasm
To run with wasm first install `trunk` like so

    cargo install --locked trunk

Then start a local dev server with

   trunk serve

To deploy to a web server just do

    trunk build --release
   

## author
vidstige
