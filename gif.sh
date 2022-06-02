#!/bin/sh
RESOLUTION=${RESOLUTION:-506x253} 
FPS=${FPS:-30}
ffmpeg -hide_banner -f rawvideo -pixel_format rgb32 -framerate $FPS -video_size $RESOLUTION -i - -metadata comment="vidstige" -vf "fps=$FPS,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $1
