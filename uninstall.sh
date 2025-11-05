#!/bin/sh

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
printf "Uninstalling plugin from %s\n" "$plugin_dir"
rm -rf "$plugin_dir"
rc=$?
[ $rc -ne 0 ] && exit rc
printf "Uninstallation complete. Would you like to remove the build artifacts as well? (y/n): "
read answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    cargo clean
    rc=$?
    [ $rc -ne 0 ] && exit rc
    printf "Build artifacts removed.\n"
else
    printf "Build artifacts retained.\n"
fi
