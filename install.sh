#!/bin/sh

if [ "$1" = "debug" ]; then
    flag=""
    build_type="debug"
else
    flag="--release"
    build_type="release"
fi
plugin_name="convergence"
if [ "$(uname -s)" = "Darwin" ]; then
    plugin_dir="${HOME}/Library/Audio/Plug-Ins/LV2/${plugin_name}.lv2"
    ext="dylib"
elif [ "$(uname -s)" = "Linux" ]; then
    plugin_dir="${HOME}/.lv2/${plugin_name}.lv2"
    ext="so"
else
    echo "Unsupported OS"
    exit 1
fi

# build plugin
cargo build $flag || exit 1
rc=$?
[ $rc -ne 0 ] && exit rc

# copy to LV2 directory
printf "Installing plugin to %s\n" "$plugin_dir"
mkdir -p "$plugin_dir"
cp -a target/$build_type/lib${plugin_name}."$ext" "$plugin_dir/"
cp -a convergence.lv2/*.ttl "$plugin_dir/"
