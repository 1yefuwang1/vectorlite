#!/bin/bash

for wheel in wheelhouse/vectorlite-wheel*/*.whl; do
    unziped_dir=$wheel.unzipped
    unzip $wheel -d $unziped_dir

    case "$wheel" in
        *linux*x86_64.whl)
            cp $unziped_dir/vectorlite_py/vectorlite.so bindings/nodejs/packages/vectorlite-linux-x64/src
            ;;
        *win*amd64.whl)
            cp $unziped_dir/vectorlite_py/vectorlite.dll bindings/nodejs/packages/vectorlite-win32-x64/src
            ;;
        *macosx*arm64.whl)
            cp $unziped_dir/vectorlite_py/vectorlite.dylib bindings/nodejs/packages/vectorlite-darwin-arm64/src
            ;;
        *macosx*x86_64.whl)
            cp $unziped_dir/vectorlite_py/vectorlite.dylib bindings/nodejs/packages/vectorlite-darwin-x64/src
            ;;
        *)
            echo "Unknown wheel type: $wheel"
            exit 1
            ;;
    esac    
done

