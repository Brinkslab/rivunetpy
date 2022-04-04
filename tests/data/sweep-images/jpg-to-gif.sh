DIR="16-bit-to-8-bit"
cd $DIR
ffmpeg -y -f image2 -framerate 3 -i %*.jpg -loop 2 "$DIR.gif"
cd ..
