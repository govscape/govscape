#!/usr/bin/env bash
set -e

# 1. Install Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v yum &> /dev/null; then
            # CentOS/RockyOS
            sudo yum install -y gcc gcc-c++ make
            sudo yum install -y python3.11
            sudo yum install -y poppler-utils
        else
            # Ubuntu/Debian
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
            sudo apt-get install -y poppler-utils
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install python@3.11
    else
        echo "Unsupported OS. Please install Python 3.11 manually."
        exit 1
    fi
else
    echo "Python 3.11 already installed."
fi

# 2. Install Poetry
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3.11 -
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "Poetry already installed."
fi

# 3. Run poetry install
echo "Running 'poetry install'..."
poetry env use 3.11
poetry install

# 4. Install s5cmd
sudo apt install -y pipx
pipx install s5cmd
