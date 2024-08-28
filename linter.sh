#!/bin/bash -e
#Get the first argument
if [ $# -eq 0 ]; then
    to_check="formatting"
else
    to_check=$1
fi

line_length=79

if [ $to_check == "check" ]; then
    echo "Running isort...Checking..."
    isort . --profile "black" --line-length $line_length --check-only
else
    echo "Running isort...Formatting..."
    isort . --profile "black" --line-length $line_length
fi


if [ $to_check == "check" ]; then
    echo "Running black...Checking..."
    black . --line-length $line_length --check
else
    echo "Running black...Formatting..."
    black . --line-length $line_length
fi

echo "Running flake8..."
flake8 . --max-line-length $line_length --ignore E501,E203,E266,W503,E741,W605 --exclude .git,venv,setup.py
