#!/bin/bash

function add_llvm_10_apt_source {
  curl https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
  if [[ $1 == "16.04" ]]; then
    echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-10 main" | sudo tee -a /etc/apt/sources.list
  elif [[ $1 == "18.04" ]]; then
    echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main" | sudo tee -a /etc/apt/sources.list
  fi
  sudo apt-get -qq update
}

function install_system_packages {
  sudo apt install -y graphviz libgraphviz-dev
  sudo apt install -y libllvm10 llvm-10-dev
  sudo apt install -y clang-10 libclang1-10 libclang-10-dev libclang-common-10-dev
}

function install_python_packages {
  CUDA=$1

  python3 -m pip install torch==1.5.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
  python3 -m pip install torchvision==0.6.0
  python3 -m pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  python3 -m pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  python3 -m pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  python3 -m pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  if [[ "$CUDA" != "cpu" ]]; then
    python3 -m pip install dgl-$CUDA
  else
    python3 -m pip install dgl
  fi
  python3 -m pip install tensorflow==2.2.0
}


if [ $# -eq 0 ]
  then
    echo "Usage: install_deps {cpu|cu92|cu100|cu101}"
    exit 1
fi

if [[ $(lsb_release -rs) == "16.04" ]] || [[ $(lsb_release -rs) == "18.04" ]]; then
  echo "OS supported."
  if ! grep -q 'llvm-toolchain-.*-10' /etc/apt/sources.list; then
    add_llvm_10_apt_source $(lsb_release -rs)
  fi
elif [[ $(lsb_release -rs) == "20.04" ]]; then
  echo "OS supported."
else
  echo "Non-supported OS. You have to install the packages manually."
  exit 1
fi

install_system_packages
install_python_packages $1
