#!/usr/bin/env bash
# Forked from https://github.com/sosi-org/scientific-code/blob/main/timescales-state/run-script.bash
# which was in turn forked from https://github.com/sosi-org/primsoup/blob/master/actn/run-actn.bash

set -xu

mkdir -p temp
source ./temp/my-bash-utils.sh || curl -k \
    https://raw.githubusercontent.com/sohale/implisolid/revival-sohale/scripts/bash-utils.sh \
    >./temp/my-bash-utils.sh

source ./temp/my-bash-utils.sh

set -e

export PIPFLAGS="\
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org"
echo ">>>>$PIPFLAGS"


function chk_venv(){
    # a solution based on `venv` as opposed to `virutalenv`

    #set -ex
    if [[  -d ./p3-for-me ]]
    then
    echo "venv exists"
    return 0
    fi

    echo "INSTALLING THEM"
    rm -rf p3-for-me || :

    # venv is shipped with python3
    #python3 -m venv -v --python=python3 p3-for-me
    python3 -m venv p3-for-me
    source ./p3-for-me/bin/activate

    python --version
    # Python 3.9.12

    # For trusted sources: see  https://stackoverflow.com/questions/49324802/pip-always-fails-ssl-verification

    python -m \
        pip install \
            $PIPFLAGS \
            --upgrade pip

    #python -m \
        pip install \
            $PIPFLAGS \
            numpy

        pip install \
            $PIPFLAGS \
            matplotlib

}

MAKE_HAPPEN "./p3-for-me/bin/activate" || {
chk_venv
}

source ./p3-for-me/bin/activate


MAKE_HAPPEN "./p3-for-me/lib/python3.9/site-packages/scipy/LICENSE.txt" || {
  pip install $PIPFLAGS scipy
}
MAKE_HAPPEN "./p3-for-me/lib/python3.9/site-packages/sympy/__init__.py" || {
  pip install $PIPFLAGS sympy
}
MAKE_HAPPEN "./p3-for-me/lib/python3.9/site-packages/yaml/__init__.py" || {
  pip install $PIPFLAGS PyYAML
}
#MAKE_HAPPEN "./p3-for-me/lib/python3.9/site-packages/pdb/__init__.py" || {
#  pip install $PIPFLAGS pdb
#}

# python -m pip install -U autopep8

MAKE_HAPPEN "./p3-for-me/lib/python3.9/site-packages/graphviz/__init__.py" || {
  pip install $PIPFLAGS graphviz
}

echo "Main script"

source ./p3-for-me/bin/activate

python --version

python simult.py

<<< '
source ./p3-for-me/bin/activate
python fitzhugh-nagumo-model-3.py
'