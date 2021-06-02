if [[ $# -ne 1 ]]
then
    echo "Need the suffix as argument"
    echo "Usage:"
    echo "./make_gif.sh bubu"
    exit
fi
suffix=$1

ffmpeg -i regression_%d.png -vf palettegen palette.png
ffmpeg -i regression_%d.png -i palette.png -lavfi paletteuse ../leaky_relu_${suffix}.gif

